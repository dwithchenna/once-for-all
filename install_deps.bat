@echo off
echo Installing required packages for OFA Custom Hardware...
echo.

echo Step 1: Installing core PyTorch packages...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if %errorlevel% neq 0 (
    echo Failed to install PyTorch
    exit /b 1
)

echo Step 2: Installing ONNX packages...
pip install onnx onnxruntime
if %errorlevel% neq 0 (
    echo Failed to install ONNX packages
    exit /b 1
)

echo Step 3: Installing utility packages...
pip install PyYAML tqdm numpy filelock gdown
if %errorlevel% neq 0 (
    echo Failed to install utility packages
    exit /b 1
)

echo Step 4: Installing ML packages...
pip install scikit-learn matplotlib
if %errorlevel% neq 0 (
    echo Failed to install ML packages
    exit /b 1
)

echo.
echo Installation complete!
echo Run 'python test_setup.py' to verify the installation.
pause
