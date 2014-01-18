#!/bin/sh

REVISION=$1

if test -z "$REVISION"; then
    echo "$0 REVISION"
    exit 127
fi

VERSION="r$REVISION"
echo "VERSION = $VERSION"

URL=`LANG=C svn info | grep 'URL:' | awk '{print $2}'`

DIR="looper-standalone-$VERSION"
svn export -r $REVISION $URL $DIR

# remove directories
(cd $DIR && rm -rf release.sh affinity.h bond_percolation.* communication_test.C fjsamp* loop0.* parallel_* sw_r.C union_find_r.* visual* vmpeak.C)

# make tar.bz2
tar jcf $DIR.tar.bz2 $DIR
