infinite-trailer
================

Auto-generated trailers that play randomly forever.

Basically looks for black frames in a video and cuts the trailer into clips at
those time points.

Requires:

- opencv python bindings. Download them using your system package manager
or from the OpenCV site.
- ffmpeg on your path
- youtube-dl on your path


## Create an infinite trailer

To download the trailers, scan for the black out sections, cut out the clips
and then create the listings for the web player, run

    $ python2 infinite-trailer.py bulk

Then run the web player

    $ cd html
    $ python2 -m SimpleHTTPServer

And go to [http://localhost:8000](http://localhost:8000) to view your infinite
trailer.

## How to add more trailers?

The list of trailers are in the `trailers.json`. Just add the name you want the
file to be and its youtube id.


