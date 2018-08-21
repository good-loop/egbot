#!/bin/bash

# Egbot frontend updater

############
## Variable Declarations and Arrays
############
LESSFILES=('src/style/main.less' 'src/style/print.less')
LESSOUTDIR='web/style'
SYNCTHESE=("web" "src" "package.json" "webpack.config.js")
RSYNCSHORT="rsync -rhPL"
DEST="winterwell@robinson.soda.sh:/home/winterwell/egbot.good-loop.com"
############
## Usage
############
function usage {
    printf "\nUse this tool as such:\n\t'./update.frontend.sh' --this will take whatever html,less,js, and package.json files are in your project directory, and sync them to the egbot server\n\t'./update.frontend.sh clean' -- This will perform the sync, but also clean all of the npm packages on the egbot server\n\n"
}

###########
## Handle Arguments
###########
case $1 in
    '')
        CLEAN='false'
    ;;
    clean|CLEAN)
        CLEAN='true'
    ;;
    *)
        usage
        exit 0
    ;;
esac

############
## 'Clean' function
############
function clean {
    if [[ $CLEAN = 'true' ]]; then
        printf "\n...Cleaning the node_modules directory on Robinson...\n"
        ssh winterwell@robinson.soda.sh 'rm -rf /home/winterwell/egbot.good-loop.com/node_modules'
    fi
}

############
## 01. Convert LESS files
############
for file in "${LESSFILES[@]}"; do
		if [ -e "$file" ]; then
			printf "\nconverting $file\n"
			F=`basename $file`
			printf lessc "$file" "$LESSOUTDIR/${F%.less}.css"
			lessc "$file" "$LESSOUTDIR/${F%.less}.css"
		else
			printf "\nless file not found: $file\n"				
		fi
done

#############
## 02. Perform the Sync
#############
for item in ${SYNCTHESE[@]}; do
    $RSYNCSHORT $item $DEST
done

##############
## 03. Update NPM packages and Webpack
##############
clean  # Only runs if argued
printf "\nGetting NPM Dependencies ...\n"
ssh winterwell@robinson.soda.sh 'cd /home/winterwell/egbot.good-loop.com && npm i'
printf "\nWebpacking...\n"
ssh winterwell@robinson.soda.sh 'cd /home/winterwell/egbot.good-loop.com && webpack --progress -p'
printf "\nFrontend updated\n"