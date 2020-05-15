#!/usr/bin/env sh

cp header.html hoffstuff.html && cmark --unsafe --width 80 hoffstuff.md >> hoffstuff.html
