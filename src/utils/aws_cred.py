import boto3
import os
import logging
import threading
from botocore.credentials import RefreshableCredentials
from botocore.session import get_session as get_botocore_session
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from datetime import datetime, timezone

# Load environment variables (for local dev)
load_dotenv(override=True)

logger = logging.getLogger(__name__)

# Configure boto3 with retries and timeouts
boto_config = Config(
    retries={"max_attempts": 5, "mode": "adaptive"},
    connect_timeout=5,
    read_timeout=150,
    max_pool_connections=50
)

# Global session (singleton pattern)
_global_session = None
_session_lock = threading.Lock()
_credentials_expiry = None

# Store original base credentials separately so we can always use them
# to assume a new role, even when assumed role credentials expire
_base_access_key = None
_base_secret_key = None


def assume_role():
    """
    Assumes the IAM role specified by AWS_ROLE_ARN using the base credentials
    provided via environment variables. Returns temporary STS credentials.
    
    Uses stored base credentials if available, otherwise reads from environment.
    This ensures we can always assume a new role even when assumed role credentials expire.
    """
    global _base_access_key, _base_secret_key
    
    # Use stored base credentials if available, otherwise read from environment
    # This is important because we may have overwritten env vars with assumed role credentials
    USER_ACCESS_KEY = _base_access_key or os.getenv("AWS_ACCESS_KEY_ID")
    USER_SECRET_KEY = _base_secret_key or os.getenv("AWS_SECRET_ACCESS_KEY")
    ROLE_ARN = os.getenv("AWS_ROLE_ARN")
    REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    
    # Store base credentials for future use
    if _base_access_key is None:
        _base_access_key = USER_ACCESS_KEY
    if _base_secret_key is None:
        _base_secret_key = USER_SECRET_KEY

    # Validate required env variables
    if not all([USER_ACCESS_KEY, USER_SECRET_KEY, ROLE_ARN]):
        missing = []
        if not USER_ACCESS_KEY:
            missing.append("AWS_ACCESS_KEY_ID")
        if not USER_SECRET_KEY:
            missing.append("AWS_SECRET_ACCESS_KEY")
        if not ROLE_ARN:
            missing.append("AWS_ROLE_ARN")
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    try:
        # Create STS client using base credentials (not assumed role credentials)
        sts_client = boto3.client(
            "sts",
            aws_access_key_id=USER_ACCESS_KEY,
            aws_secret_access_key=USER_SECRET_KEY,
            region_name=REGION,
            config=boto_config
        )

        response = sts_client.assume_role(
            RoleArn=ROLE_ARN,
            RoleSessionName="bedrock_access_session",
            DurationSeconds=21600  # 6 hours
        )

        credentials = response["Credentials"]
        logger.info(f"✅ Assumed AWS Role Successfully! Expires at: {credentials['Expiration']}")

        return credentials

    except Exception as e:
        logger.error(f"❌ Failed to assume role: {e}", exc_info=True)
        raise RuntimeError(f"Failed to assume AWS role: {e}") from e


def _is_token_expired_error(error):
    """
    Check if the error is related to expired credentials/tokens.
    """
    error_message = str(error).lower()
    expired_keywords = [
        'expired',
        'expiredtoken',
        'expiredtokenexception',
        'token has expired',
        'security token included in the request is expired',
        'security token',
        'credentials have expired',
        'accessdenied',
        'invalid security token'
    ]
    return any(keyword in error_message for keyword in expired_keywords)


def _refresh_credentials():
    """
    Refresh function called automatically by botocore when credentials expire.
    Must return a dict with: access_key, secret_key, token, expiry_time
    
    Also exposes credentials as environment variables so all boto3 clients
    (including LangChain internals) use the assumed role automatically.
    """
    global _credentials_expiry
    
    logger.info("🔄 Refreshing AWS credentials...")
    credentials = assume_role()
    
    # Track expiry time
    _credentials_expiry = credentials["Expiration"]
    
    # Expose assumed role credentials as env vars so all boto3 clients
    # (including LangChain internals) use the assumed role automatically.
    os.environ["AWS_ACCESS_KEY_ID"] = credentials["AccessKeyId"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = credentials["SecretAccessKey"]
    os.environ["AWS_SESSION_TOKEN"] = credentials["SessionToken"]
    
    return {
        "access_key": credentials["AccessKeyId"],
        "secret_key": credentials["SecretAccessKey"],
        "token": credentials["SessionToken"],
        "expiry_time": credentials["Expiration"].isoformat(),
    }


def _create_refreshable_session():
    """
    Creates a boto3 session with auto-refreshing credentials using RefreshableCredentials.
    This handles credential expiration automatically without manual tracking.
    
    Returns:
        boto3.Session: Session with auto-refreshing credentials
    """
    REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    
    # Create refreshable credentials
    session_credentials = RefreshableCredentials.create_from_metadata(
        metadata=_refresh_credentials(),
        refresh_using=_refresh_credentials,
        method="sts-assume-role"
    )
    
    # Create a botocore session and attach the refreshable credentials
    botocore_session = get_botocore_session()
    botocore_session._credentials = session_credentials
    botocore_session.set_config_variable("region", REGION)
    
    # Create boto3 session from botocore session
    autorefresh_session = boto3.Session(botocore_session=botocore_session)
    
    logger.info("✅ Created auto-refreshing boto3 session")
    return autorefresh_session


def get_session():
    """
    Get or create the global auto-refreshing boto3 session (singleton pattern).
    This session automatically refreshes credentials when they expire.
    Thread-safe implementation.
    
    Returns:
        boto3.Session: Auto-refreshing session
    """
    global _global_session
    
    with _session_lock:
        if _global_session is None:
            logger.info("🔧 Initializing global boto3 session...")
            _global_session = _create_refreshable_session()
        
        return _global_session


def refresh_session():
    """
    Force refresh the global session.
    Called when credentials expire during API calls (try-catch pattern).
    """
    global _global_session
    
    with _session_lock:
        logger.info("🔄 Force refreshing global session due to expired credentials...")
        _global_session = _create_refreshable_session()
        logger.info("✅ Global session refreshed successfully")
    
    return _global_session


def call_cred(retry_count=0, max_retries=2):
    """
    Returns a bedrock-runtime client with auto-refreshing credentials.
    Includes try-catch retry logic for expired token errors.
    
    Args:
        retry_count: Current retry attempt (internal use)
        max_retries: Maximum retry attempts
    
    Returns:
        boto3.client: Bedrock runtime client with auto-refreshing credentials
    """
    try:
        session = get_session()
        bedrock_client = session.client(
            "bedrock-runtime",
            config=boto_config
        )
        
        logger.debug("✅ Bedrock Runtime client created")
        return bedrock_client
    
    except ClientError as ce:
        # Check if this is a credential expiration error
        if _is_token_expired_error(ce) and retry_count < max_retries:
            logger.warning(
                f"⚠️ Credentials expired while creating client. "
                f"Refreshing and retrying... (Attempt {retry_count + 1}/{max_retries})"
            )
            logger.warning(f"Error details: {str(ce)}")
            
            # Force refresh the session
            try:
                # Get fresh credentials and expose them as env vars
                # This ensures LangChain internals (which may create their own boto3 clients)
                # use the assumed role credentials automatically
                credentials = assume_role()
                os.environ["AWS_ACCESS_KEY_ID"] = credentials["AccessKeyId"]
                os.environ["AWS_SECRET_ACCESS_KEY"] = credentials["SecretAccessKey"]
                os.environ["AWS_SESSION_TOKEN"] = credentials["SessionToken"]
                
                # Refresh the session with new credentials
                refresh_session()
                logger.info("🔄 Credentials refreshed successfully and exposed as environment variables")
                
                # Retry with new credentials
                return call_cred(retry_count + 1, max_retries)
                
            except Exception as refresh_error:
                logger.error(f"❌ Failed to refresh credentials: {str(refresh_error)}")
                raise
        else:
            # Either not a token expiry error, or we've exhausted retries
            if retry_count >= max_retries:
                logger.error(f"❌ Max retries ({max_retries}) exceeded for credential refresh")
            logger.error(f"❌ AWS ClientError: {str(ce)}")
            raise
    
    except Exception as e:
        logger.error(f"❌ Failed to create bedrock-runtime client: {e}", exc_info=True)
        raise


def call_cred_br(retry_count=0, max_retries=2):
    """
    Returns a regular bedrock client (non-runtime) with auto-refreshing credentials.
    Includes try-catch retry logic for expired token errors.
    
    Args:
        retry_count: Current retry attempt (internal use)
        max_retries: Maximum retry attempts
    
    Returns:
        boto3.client: Bedrock client with auto-refreshing credentials
    """
    try:
        session = get_session()
        bedrock_client = session.client(
            "bedrock",
            config=boto_config
        )
        
        logger.debug("✅ Bedrock client created")
        return bedrock_client
    
    except ClientError as ce:
        # Check if this is a credential expiration error
        if _is_token_expired_error(ce) and retry_count < max_retries:
            logger.warning(
                f"⚠️ Credentials expired while creating client. "
                f"Refreshing and retrying... (Attempt {retry_count + 1}/{max_retries})"
            )
            
            # Force refresh the session
            try:
                refresh_session()
                logger.info("✅ Successfully refreshed credentials")
                
                # Retry with new credentials
                return call_cred_br(retry_count + 1, max_retries)
                
            except Exception as refresh_error:
                logger.error(f"❌ Failed to refresh credentials: {str(refresh_error)}")
                raise
        else:
            if retry_count >= max_retries:
                logger.error(f"❌ Max retries ({max_retries}) exceeded")
            logger.error(f"❌ AWS ClientError: {str(ce)}")
            raise
    
    except Exception as e:
        logger.error(f"❌ Failed to create bedrock client: {e}", exc_info=True)
        raise


def call_cred_rekognition(retry_count=0, max_retries=2):
    """
    Returns a Rekognition client with auto-refreshing credentials.
    Includes try-catch retry logic for expired token errors.
    
    Args:
        retry_count: Current retry attempt (internal use)
        max_retries: Maximum retry attempts
    
    Returns:
        boto3.client: Rekognition client with auto-refreshing credentials
    """
    try:
        session = get_session()
        rekognition_client = session.client(
            "rekognition",
            config=boto_config
        )
        
        logger.debug("✅ Rekognition client created")
        return rekognition_client
    
    except ClientError as ce:
        # Check if this is a credential expiration error
        if _is_token_expired_error(ce) and retry_count < max_retries:
            logger.warning(
                f"⚠️ Credentials expired while creating Rekognition client. "
                f"Refreshing and retrying... (Attempt {retry_count + 1}/{max_retries})"
            )
            
            # Force refresh the session
            try:
                refresh_session()
                logger.info("✅ Successfully refreshed credentials")
                
                # Retry with new credentials
                return call_cred_rekognition(retry_count + 1, max_retries)
                
            except Exception as refresh_error:
                logger.error(f"❌ Failed to refresh credentials: {str(refresh_error)}")
                raise
        else:
            if retry_count >= max_retries:
                logger.error(f"❌ Max retries ({max_retries}) exceeded")
            logger.error(f"❌ AWS ClientError: {str(ce)}")
            raise
    
    except Exception as e:
        logger.error(f"❌ Failed to create Rekognition client: {e}", exc_info=True)
        raise


def refresh_bedrock_client():
    """
    Force-refresh credentials and return a new bedrock-runtime client.
    Deprecated: Use call_cred() which handles retries automatically.
    """
    logger.warning("refresh_bedrock_client() is deprecated. Use call_cred() instead.")
    refresh_session()
    return call_cred()


def refresh_bedrock_client_br():
    """
    Force-refresh credentials and return a new bedrock client.
    Deprecated: Use call_cred_br() which handles retries automatically.
    """
    logger.warning("refresh_bedrock_client_br() is deprecated. Use call_cred_br() instead.")
    refresh_session()
    return call_cred_br()


def refresh_rekognition_client():
    """
    Force-refresh credentials and return a new Rekognition client.
    Deprecated: Use call_cred_rekognition() which handles retries automatically.
    """
    logger.warning("refresh_rekognition_client() is deprecated. Use call_cred_rekognition() instead.")
    refresh_session()
    return call_cred_rekognition()


def get_credentials_expiry():
    """
    Get the current credentials expiry time.
    Useful for health checks and monitoring.
    
    Returns:
        datetime: Expiry time of current credentials
    """
    global _credentials_expiry
    return _credentials_expiry


# ✅ Optional local test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    print("\n" + "="*60)
    print("TEST 1: Initial client creation")
    print("="*60)
    client1 = call_cred()
    
    print("\n" + "="*60)
    print("TEST 2: Reuse session (same credentials)")
    print("="*60)
    client2 = call_cred()
    
    print("\n" + "="*60)
    print("TEST 3: Create Rekognition client")
    print("="*60)
    rekognition_client = call_cred_rekognition()
    
    print("\n" + "="*60)
    print("TEST 4: Credentials expiry check")
    print("="*60)
    expiry = get_credentials_expiry()
    if expiry:
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        time_left = (expiry - now).total_seconds() / 60
        print(f"Credentials expire in: {time_left:.2f} minutes")
    
    print("\n" + "="*60)
    print("Caller Identity Check")
    print("="*60)
    sts_client = boto3.client("sts")
    identity = sts_client.get_caller_identity()
    print(f"✅ Account: {identity['Account']}")
    print(f"✅ UserId: {identity['UserId']}")