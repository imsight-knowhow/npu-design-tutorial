DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" 
echo "Executing $DIR/_custom-on-build.sh" 
bash $DIR/../../stage-2/system/pixi/install-pixi.bash --user me --cache-dir /soft/app/pixi-cache --pypi-repo tuna --conda-repo tuna
bash $DIR/../../stage-1/system/uv/install-uv.sh --pypi-repo tuna --user me
bash $DIR/../../stage-2/system/nodejs/install-nvm-nodejs.sh --with-cn-mirror --user me
bash $DIR/../../stage-2/system/codex-cli/install-codex-cli.sh --user me
bash $DIR/../../stage-2/system/claude-code/install-claude-code.sh --user me