#!/usr/bin/env python2.7

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from argparse import ArgumentParser
from contextlib import contextmanager
from time import sleep
import multiprocessing
import os
import subprocess
import sys

import cv2
import numpy


def main(argv=None):
    args = parse_args(argv=argv)
    with video_capture(args.video_path) as cap:
        scene_splitter = SceneSplitter(cap, args.threshold)
        scenes = scene_splitter.find_scenes()
    cv2.destroyAllWindows()
    render_scenes(args.video_path, scenes)


def parse_args(argv=None):
    if argv is None:
        argv = sys.argv
    parser = ArgumentParser()
    parser.add_argument('-t', '--threshold', default=10, type=int)
    parser.add_argument('-l', '--max-length', default=5, type=int)
    parser.add_argument('-v', '--show-video', action='store_true',
                        default=False)
    parser.add_argument('video_path', help='video file to create clips from')
    return parser.parse_args(args=argv[1:])


@contextmanager
def video_capture(vido_path):
    cap = cv2.VideoCapture(vido_path)
    yield cap
    cap.release()


class SceneSplitter(object):
    def __init__(self, cap, threshold):
        self._cap = cap
        self._threshold = threshold

        self._find_scenes_called = False

        self._in_fade = False
        self._scenes = []
        self._start_index = 0

        self._video_width = self._get_int_prop('FRAME_WIDTH')
        self._video_height = self._get_int_prop('FRAME_HEIGHT')
        self._video_fps = self._get_int_prop('FPS')

    def _get_int_prop(self, prop_name):
        name = 'CV_CAP_PROP_{prop_name}'.format(prop_name=prop_name)
        return int(self._cap.get(getattr(cv2.cv, name)))

    def find_scenes(self):
        if not self._find_scenes_called:
            self._find_scenes_called = True
            for index, frame in enumerate(self._frames()):
                self._check_frame(index, frame)
        return self._scenes

    def _frames(self):
        while True:
            ret, frame = self._cap.read()
            if not ret:
                raise StopIteration
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _check_frame(self, index, frame):
        if self._count_light_pixels(frame) == 0:
            if not self._in_fade:
                self._in_fade = True
                self._add_frame(self._start_index, index)
        elif self._in_fade:
            self._in_fade = False
            self._start_index = index

    def _count_light_pixels(self, frame):
        return numpy.count_nonzero(frame > self._threshold)

    def _add_frame(self, start_index, stop_index):
        def timestamp(index):
            return index / self._video_fps
        scene = (timestamp(start_index), timestamp(stop_index))
        print(scene)
        self._scenes.append(scene)


def render_scenes(video_path, scenes):
    pool = multiprocessing.Pool()
    for index, (start_time, stop_time) in enumerate(scenes):
        pool.apply_async(render_scene, [
            index,
            video_path,
            start_time,
            stop_time
        ])
    pool.close()
    pool.join()


def render_scene(index, video_path, start_time, stop_time):
    path, ext = os.path.splitext(video_path)
    path = '{path}-{index}.ogg'.format(path=path, index=index)
    if os.path.isfile(path):
        os.remove(path)

    args = [
        '/usr/bin/ffmpeg',
        '-i', video_path,
        '-q', '5',
        '-pix_fmt', 'yuv420p',
        '-acodec', 'libvorbis',
        '-vcodec', 'libtheora',
        '-ss', str(start_time),
        '-to', str(stop_time),
        path,
    ]

    subprocess.check_call(args)


if __name__ == '__main__':
    sys.exit(main())

