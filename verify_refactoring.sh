#!/bin/bash
################################################################################
# Refactoring Verification Script
# 
# This script verifies that the refactored codebase is correctly structured
# and ready for use.
################################################################################

set -e  # Exit on error

echo "================================================================================"
echo "REFACTORING VERIFICATION"
echo "================================================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

success() {
    echo -e "${GREEN}✓${NC} $1"
}

warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

error() {
    echo -e "${RED}✗${NC} $1"
}

# Change to project directory
cd /n/holylabs/kempner_dev/Users/hsafaai/Code/vine_diffusion_copula

echo "1. Checking file structure..."
echo "----------------------------------------"

# Check core library files
if [ -f "vdc/train/unified_trainer.py" ]; then
    success "vdc/train/unified_trainer.py exists"
else
    error "vdc/train/unified_trainer.py missing"
    exit 1
fi

if [ -f "vdc/train/__init__.py" ]; then
    success "vdc/train/__init__.py exists"
else
    error "vdc/train/__init__.py missing"
    exit 1
fi

if [ -f "scripts/train_unified.py" ]; then
    success "scripts/train_unified.py exists"
else
    error "scripts/train_unified.py missing"
    exit 1
fi

if [ -f "vdc/vine/copula_diffusion.py" ]; then
    success "vdc/vine/copula_diffusion.py exists"
else
    warning "vdc/vine/copula_diffusion.py missing (should exist from previous work)"
fi

echo ""
echo "2. Checking configurations..."
echo "----------------------------------------"

ACTIVE_CONFIGS=(
    "configs/validate_no_probit_diffusion_m128.yaml"
    "configs/train_diffusion_uniform_m128.yaml"
    "configs/train_diffusion_probit_m128.yaml"
    "configs/train_diffusion_boundary_m128.yaml"
)

for config in "${ACTIVE_CONFIGS[@]}"; do
    if [ -f "$config" ]; then
        success "$(basename $config)"
    else
        error "$(basename $config) missing"
    fi
done

echo ""
echo "3. Checking SLURM scripts..."
echo "----------------------------------------"

ACTIVE_SLURM=(
    "slurm_jobs/validate_diffusion_no_probit_m128.sh"
    "slurm_jobs/train_diffusion_uniform_m128.sh"
    "slurm_jobs/train_diffusion_probit_m128.sh"
    "slurm_jobs/train_diffusion_boundary_m128.sh"
)

for script in "${ACTIVE_SLURM[@]}"; do
    if [ -f "$script" ]; then
        if [ -x "$script" ]; then
            success "$(basename $script) (executable)"
        else
            warning "$(basename $script) (not executable)"
            chmod +x "$script"
            success "  → Made executable"
        fi
    else
        error "$(basename $script) missing"
    fi
done

echo ""
echo "4. Checking archive..."
echo "----------------------------------------"

if [ -d "archive" ]; then
    success "archive/ directory exists"
    
    archived_configs=$(find configs/archive -name "*.yaml" 2>/dev/null | wc -l)
    success "  - Configs archived: $archived_configs"
    
    archived_slurm=$(find slurm_jobs/archive -name "*.sh" 2>/dev/null | wc -l)
    success "  - SLURM scripts archived: $archived_slurm"
    
    archived_scripts=$(find archive/scripts -name "*.py" 2>/dev/null | wc -l)
    success "  - Scripts archived: $archived_scripts"
else
    warning "archive/ directory doesn't exist"
fi

echo ""
echo "5. Checking .gitignore..."
echo "----------------------------------------"

if grep -q "^archive/" .gitignore 2>/dev/null; then
    success "archive/ is gitignored"
else
    warning "archive/ not in .gitignore"
fi

if grep -q "^results/" .gitignore 2>/dev/null; then
    success "results/ is gitignored"
else
    warning "results/ not in .gitignore"
fi

echo ""
echo "6. Checking documentation..."
echo "----------------------------------------"

DOCS=(
    "REFACTORING_SUMMARY.txt"
    "IMPLEMENTATION_COMPLETE.txt"
    "QUICK_START.md"
    "CHANGES.md"
)

for doc in "${DOCS[@]}"; do
    if [ -f "$doc" ]; then
        success "$doc"
    else
        warning "$doc missing"
    fi
done

echo ""
echo "7. Python syntax check..."
echo "----------------------------------------"

if command -v python3 &> /dev/null; then
    if python3 -m py_compile vdc/train/unified_trainer.py 2>/dev/null; then
        success "vdc/train/unified_trainer.py syntax OK"
    else
        error "vdc/train/unified_trainer.py has syntax errors"
    fi
    
    if python3 -m py_compile scripts/train_unified.py 2>/dev/null; then
        success "scripts/train_unified.py syntax OK"
    else
        error "scripts/train_unified.py has syntax errors"
    fi
else
    warning "python3 not available, skipping syntax check"
fi

echo ""
echo "8. Line count comparison..."
echo "----------------------------------------"

if [ -f "archive/scripts/train_unified_BACKUP.py" ] && [ -f "scripts/train_unified.py" ]; then
    old_lines=$(wc -l < archive/scripts/train_unified_BACKUP.py)
    new_lines=$(wc -l < scripts/train_unified.py)
    lib_lines=$(wc -l < vdc/train/unified_trainer.py)
    
    reduction=$((100 - (new_lines * 100 / old_lines)))
    
    echo "  Original script:    $old_lines lines"
    echo "  New CLI wrapper:    $new_lines lines"
    echo "  Library module:     $lib_lines lines"
    success "CLI size reduction: $reduction%"
else
    warning "Cannot compare line counts (backup missing)"
fi

echo ""
echo "================================================================================"
echo "SUMMARY"
echo "================================================================================"
echo ""
echo "The refactoring verification is complete."
echo ""
echo "Next steps:"
echo ""
echo "  1. Test the refactored training:"
echo "     sbatch slurm_jobs/validate_diffusion_no_probit_m128.sh"
echo ""
echo "  2. Try the boundary-focused config:"
echo "     sbatch slurm_jobs/train_diffusion_boundary_m128.sh"
echo ""
echo "  3. Use the API in your code:"
echo "     from vdc.vine.copula_diffusion import DiffusionCopulaModel"
echo "     model = DiffusionCopulaModel.from_checkpoint('path/to/checkpoint.pt')"
echo ""
echo "For detailed documentation, see:"
echo "  - QUICK_START.md (quick reference)"
echo "  - REFACTORING_SUMMARY.txt (technical details)"
echo "  - IMPLEMENTATION_COMPLETE.txt (complete feature list)"
echo "  - CHANGES.md (change log)"
echo ""
echo "================================================================================"

