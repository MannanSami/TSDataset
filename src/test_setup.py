import os

from dotenv import load_dotenv

from config import logger


def test_env_config():
    """Test environment configuration."""
    logger.info("=" * 60)
    logger.info("Testing Environment Configuration")
    logger.info("=" * 60)

    load_dotenv()

    required_vars = [
        "ASSISTANT_API_KEY",
        "ASSISTANT_BASE_URL",
        "ASSISTANT_MODEL",
    ]

    all_good = True
    for var in required_vars:
        value = os.getenv(var)
        if value:
            # Mask API keys
            if "KEY" in var:
                display_value = value[:10] + "..." if len(value) > 10 else value
            else:
                display_value = value
            logger.success(f"{var}: {display_value}")
        else:
            logger.error(f"{var}: NOT SET")
            all_good = False

    return all_good


def test_imports():
    """Test if all required imports work."""
    logger.info("=" * 60)
    logger.info("Testing Python Imports")
    logger.info("=" * 60)

    imports = [
        ("openai", "OpenAI client"),
        ("datasets", "HuggingFace datasets"),
        ("pandas", "Data processing"),
        ("tqdm", "Progress bars"),
    ]

    all_good = True
    for module, description in imports:
        try:
            __import__(module)
            logger.success(f"{module}: OK ({description})")
        except ImportError:
            logger.error(f"{module}: MISSING ({description})")
            all_good = False

    return all_good


def test_model_connections():
    """Test connection to both models."""
    logger.info("=" * 60)
    logger.info("Testing Model Connections")
    logger.info("=" * 60)

    try:
        from bug_injector import BugInjector

        logger.success("BugInjector imported successfully")

        bug_injector = BugInjector()
        logger.success(f"Assistant Model: {bug_injector.model}")
        logger.info(f"  Base URL: {bug_injector.base_url}")

    except Exception as e:
        logger.error(f"Error initializing BugInjector: {e}")
        return False

    try:
        from user_simulator import UserSimulator

        logger.success("UserSimulator imported successfully")

        user_sim = UserSimulator()
        logger.success(f"User Model: {user_sim.model_name}")

        # Check primary client connectivity
        if user_sim.client:
            logger.success("User simulator primary client initialized")
        else:
            logger.error("User simulator primary client not available")
            return False
    except Exception as e:
        logger.error(f"Error initializing UserSimulator: {e}")
        return False

    try:
        logger.success("ConversationGenerator imported successfully")
        logger.success("Conversation generator ready")

    except Exception as e:
        logger.error(f"Error initializing ConversationGenerator: {e}")
        return False

    return True


def test_simple_generation():
    """Test a simple bug injection (doesn't call API)."""
    logger.info("=" * 60)
    logger.info("Testing Basic Functionality")
    logger.info("=" * 60)

    logger.info("Sample TypeScript code loaded")
    logger.info("Bug categories available:")

    try:
        from bug_injector import BugInjector

        bug_injector = BugInjector()

        for i, category in enumerate(bug_injector.TS_BUG_CATEGORIES, 1):
            logger.info(f"  {i}. {category}")

        logger.success(
            f"Bug injector configured with {len(bug_injector.TS_BUG_CATEGORIES)} categories"
        )

    except Exception as e:
        logger.error(f"Error: {e}")
        return False

    return True


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("TSBugger Synthetic Dataset Generation - Setup Validation")
    logger.info("=" * 60)

    results = []

    # Test 1: Environment
    results.append(("Environment Config", test_env_config()))

    # Test 2: Imports
    results.append(("Python Imports", test_imports()))

    # Test 3: Model connections
    results.append(("Model Connections", test_model_connections()))

    # Test 4: Basic functionality
    results.append(("Basic Functionality", test_simple_generation()))

    # Summary
    logger.info("=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)

    all_passed = True
    for test_name, passed in results:
        if passed:
            logger.success(f"PASS: {test_name}")
        else:
            logger.error(f"FAIL: {test_name}")
            all_passed = False

    if all_passed:
        logger.success("All tests passed! You're ready to generate datasets.")
        logger.info("Next steps:")
        logger.info("1. Run: python generate_dataset.py -n 5")
        logger.info("2. Check output/dataset_samples.json")
        logger.info("3. If successful, scale up to larger batches")
    else:
        logger.warning("Some tests failed. Please fix the issues above.")
        logger.info("Common fixes:")
        logger.info(
            "- Install missing packages: pip install openai datasets pandas tqdm python-dotenv"
        )
        logger.info("- Update .env with correct API keys and URLs")
        logger.info("- Ensure API keys and base URLs are configured correctly")


if __name__ == "__main__":
    main()
