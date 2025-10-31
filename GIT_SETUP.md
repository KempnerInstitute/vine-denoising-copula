# Git Repository Setup

This project uses **two separate git repositories**:

## 📁 Repository Structure

### 1. Main Codebase Repository
- **Location**: `/n/holylabs/kempner_dev/Users/hsafaai/Code/vine_diffusion_copula/`
- **Remote**: `git@github.com:KempnerInstitute/vine-diffusion-copula.git`
- **Purpose**: Main vine copula implementation code
- **Excludes**: `drafts/` folder (in .gitignore)

### 2. Paper Drafts Repository
- **Location**: `/n/holylabs/kempner_dev/Users/hsafaai/Code/vine_diffusion_copula/drafts/`
- **Remote**: `git@github.com:houman1359/diffusion-vine-copula.git`
- **Purpose**: ICML 2026 paper drafts and documentation
- **Independent**: Separate git history from main repo

---

## 🚀 Quick Commands

### Working with Main Repository (Codebase)
```bash
cd /n/holylabs/kempner_dev/Users/hsafaai/Code/vine_diffusion_copula

# Check status
git status

# Add files
git add .

# Commit
git commit -m "Your commit message"

# Push to KempnerInstitute repo
git push origin master  # or 'main' if you renamed the branch

# Pull latest changes
git pull origin master
```

### Working with Drafts Repository (Paper)
```bash
cd /n/holylabs/kempner_dev/Users/hsafaai/Code/vine_diffusion_copula/drafts

# Check status
git status

# Add files
git add .

# Commit
git commit -m "Update paper draft"

# Push to houman1359 repo
git push origin master  # or 'main' if you renamed the branch

# Pull latest changes
git pull origin master
```

---

## 📝 Initial Commit Steps

### For Main Repository
```bash
cd /n/holylabs/kempner_dev/Users/hsafaai/Code/vine_diffusion_copula

# Stage all files
git add .

# Create initial commit
git commit -m "Initial commit: Vine copula implementation"

# Rename branch to 'main' (optional but recommended)
git branch -M main

# Push to GitHub
git push -u origin main
```

### For Drafts Repository
```bash
cd /n/holylabs/kempner_dev/Users/hsafaai/Code/vine_diffusion_copula/drafts

# Stage all files
git add .

# Create initial commit
git commit -m "Initial commit: ICML 2026 paper draft"

# Rename branch to 'main' (optional)
git branch -M main

# Push to GitHub
git push -u origin main
```

---

## ⚠️ Important Notes

1. **Separate Histories**: These two repos have completely independent git histories
2. **No Submodule**: The drafts folder is NOT a git submodule, just a separate repo
3. **Gitignore**: The main repo's `.gitignore` excludes `drafts/` to prevent conflicts
4. **Two Remotes**: Always check which directory you're in before committing!

---

## 🔧 Troubleshooting

### Check which repo you're in:
```bash
pwd                    # Show current directory
git remote -v         # Show which remote this repo connects to
```

### If you accidentally commit to wrong repo:
```bash
git reset --soft HEAD~1   # Undo last commit (keeps changes staged)
```

### To change branch name from 'master' to 'main':
```bash
git branch -M main
git push -u origin main
```

---

## 📊 Current Status

Both repositories are initialized and connected:

✅ **Main Repo**: Connected to `git@github.com:KempnerInstitute/vine-diffusion-copula.git`
✅ **Drafts Repo**: Connected to `git@github.com:houman1359/diffusion-vine-copula.git`
✅ **Gitignore**: `drafts/` excluded from main repo

Ready for initial commits! 🎉
