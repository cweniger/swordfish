#!/bin/sh

echo "Clone gh-pages."
git clone -b gh-pages "https://cweniger@github.com/cweniger/swordfish.git" deploy > /dev/null 2>&1 || exit 1
cd deploy

echo "Update documentation."
git rm * -rf > /dev/null 2>&1 || exit 1
cp ../html/swordfish/* ./
git add *
git commit -m "Deploy documentation" > /dev/null 2>&1 || exit 1
git push origin gh-pages > /dev/null 2>&1 || exit 1
rm deploy -rf

echo "Pushed updated documentation."
exit 0
