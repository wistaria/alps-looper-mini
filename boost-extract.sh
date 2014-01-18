#!/bin/sh

BCP="$1"
BOOST_DIR="$2"

mkdir -p boost
rm -rf boost/*
$BCP --scan --boost=$BOOST_DIR *.C *.h boost
