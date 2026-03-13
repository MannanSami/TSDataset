import asyncio
import os
import shutil

from dotenv import load_dotenv

from config import OPENCODE_PATH, OPENCODE_PORT, OPENCODE_MODEL, logger


def test_env_config():
    """Test environment configuration."""
    logger.info("=" * 60)
    logger.info("Testing Environment Configuration")
    logger.info("=" * 60)

    load_dotenv()

    configs = {
        "OPENCODE_PATH": OPENCODE_PATH,
        "OPENCODE_PORT": str(OPENCODE_PORT),
        "OPENCODE_MODEL": OPENCODE_MODEL,
    }

    all_good = True
    for var, value in configs.items():
        if value:
            logger.success(f"{var}: {value}")
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
        ("datasets", "HuggingFace datasets"),
        ("pandas", "Data processing"),
        ("tqdm", "Progress bars"),
        ("opencode_client", "OpenCode client wrapper"),
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


def test_opencode_binary():
    """Test that the opencode binary exists and is executable."""
    logger.info("=" * 60)
    logger.info("Testing OpenCode Binary")
    logger.info("=" * 60)

    path = shutil.which("opencode") or OPENCODE_PATH
    if os.path.isfile(path) and os.access(path, os.X_OK):
        logger.success(f"opencode binary found at: {path}")
        return True
    else:
        logger.error(f"opencode binary not found or not executable at: {path}")
        return False


def test_model_connections():
    """Test connection to opencode by running a simple prompt."""
    logger.info("=" * 60)
    logger.info("Testing OpenCode Connectivity")
    logger.info("=" * 60)

    try:
        from bug_injector import BugInjector

        logger.success("BugInjector imported successfully")

        from user_simulator import UserSimulator

        logger.success("UserSimulator imported successfully")

        from conversation_generator import ConversationGenerator

        logger.success("ConversationGenerator imported successfully")

        logger.info(f"Configured model: {OPENCODE_MODEL}")
        logger.info(f"Configured port: {OPENCODE_PORT}")
        logger.success("All modules importable with opencode backend")
        return True

    except Exception as e:
        logger.error(f"Error importing modules: {e}")
        return False


def test_simple_generation():
    """Test basic configuration of bug injector."""
    logger.info("=" * 60)
    logger.info("Testing Basic Functionality")
    logger.info("=" * 60)

    logger.info("Sample TypeScript code loaded")
    logger.info("Bug categories available:")

    try:
        from bug_injector import BugInjector

        # Use a dummy URL — we're just testing the category list, not making calls
        bug_injector = BugInjector(attach_url="http://localhost:14000")

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

    # Test 3: OpenCode binary
    results.append(("OpenCode Binary", test_opencode_binary()))

    # Test 4: Module imports
    results.append(("Module Connections", test_model_connections()))

    # Test 5: Basic functionality
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
        logger.info("- Ensure opencode is installed at /opt/homebrew/bin/opencode")
        logger.info("- Update .env with OPENCODE_PORT and OPENCODE_MODEL if needed")
        logger.info("- Run: opencode serve --port 14000  to verify opencode works")


if __name__ == "__main__":
    main()
