#!/usr/bin/env zsh

setopt ERR_EXIT
setopt NO_UNSET

repo=$(realpath "$(dirname "$(realpath -- $0)")")
trailers_dir=$repo/trailers
scenes_dir=$repo/scenes
html_dir=$repo/html
clips_dir=$html_dir/clips
listing_path=$html_dir/listings.json

function it()
{
    python2 $repo/infinite_trailer.py $@
}

for trailer in $trailers_dir/*.mp4; do
    it find $trailer $scenes_dir
    it render $scenes_dir/${trailer:r:t}.json $trailer $clips_dir
done

it listing $html_dir $listing_path

