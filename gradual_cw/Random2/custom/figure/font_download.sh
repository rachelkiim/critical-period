#!/bin/bash
set -e

echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" \
  | sudo debconf-set-selections

sudo apt update
sudo apt install -y ttf-mscorefonts-installer unzip wget

fc-cache -f -v

fc-list | grep -Ei "Arial"
rm -f ~/.cache/matplotlib/fontlist-v*.json
rm -rf ~/.cache/matplotlib
python -u -c "import matplotlib.font_manager as fm; print('Arial:', any('Arial' in f.name for f in fm.fontManager.ttflist))"
