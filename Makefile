# Detect OS - Windows first since uname won't exist there
ifeq ($(OS),Windows_NT)
    PLATFORM := Windows
    BUILD_CMD := uv run python -m nuitka --standalone \
		src/main.py \
		--include-package=src \
		--enable-plugin=pyside6 \
		--windows-disable-console \
		--windows-icon-from-ico=icon.ico \
		--output-filename=FunscriptFlow.exe \
		--output-dir=dist
    BUILD_MSG := Building Windows executable...
    BUILD_SUCCESS := Build complete! Executable created in dist/
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Darwin)
        PLATFORM := macOS
        BUILD_CMD := uv run python -m nuitka --standalone \
			src/main.py \
			--include-package=src \
			--enable-plugin=pyside6 \
			--macos-create-app-bundle \
			--macos-app-icon=icon.png \
			--macos-app-name="Funscript Flow" \
			--output-dir=dist
        BUILD_MSG := Building macOS app bundle...
        BUILD_SUCCESS := Build complete! App bundle created in dist/
    else ifeq ($(UNAME_S),Linux)
        PLATFORM := Linux
        BUILD_CMD := uv run python -m nuitka --standalone \
			src/main.py \
			--include-package=src \
			--enable-plugin=pyside6 \
			--linux-icon=icon.png \
			--output-filename=FunscriptFlow \
			--output-dir=dist
        BUILD_MSG := Building Linux executable...
        BUILD_SUCCESS := Build complete! Executable created in dist/
    else
        PLATFORM := Unknown
        BUILD_CMD := echo "Error: Unsupported platform"
        BUILD_MSG := Error: Unsupported platform
        BUILD_SUCCESS := Build failed
    endif
endif

.PHONY: help build clean install run

help:
	@echo "Funscript Flow - Build Commands"
	@echo "==============================="
	@echo "make build    - Build for current platform ($(PLATFORM))"
	@echo "make clean    - Remove build artifacts"
	@echo "make install  - Install dependencies"
	@echo "make run      - Run application in development mode"
	@echo "make help     - Show this help message"

# Install dependencies
install:
	@echo "Installing dependencies..."
	uv sync --extra build

# Run application in development mode
run:
	@echo "Running Funscript Flow..."
	uv run python -m src.main

# Build for current platform
build: install
	@echo "$(BUILD_MSG)"
	@$(BUILD_CMD)
	@echo "$(BUILD_SUCCESS)"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf dist/
	rm -rf build/
	rm -rf main.build/
	rm -rf main.dist/
	rm -rf main.onefile-build/
	rm -rf FunscriptFlow.build/
	rm -rf FunscriptFlow.dist/
	rm -rf FunscriptFlow.onefile-build/
	rm -rf *.app
	rm -rf *.exe
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean complete!"