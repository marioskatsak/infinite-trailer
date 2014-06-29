#!/usr/bin/env python2.7

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from argparse import ArgumentParser
from collections import OrderedDict
from contextlib import contextmanager
import json
import logging
import multiprocessing
import os
import shutil
import subprocess
import sys

import cv2
import numpy


LOG_LEVELS = (
    logging.CRITICAL,
    logging.ERROR,
    logging.WARNING,
    logging.INFO,
    logging.DEBUG
)

LOG_LEVEL_TO_NAMES = OrderedDict((level, logging.getLevelName(level).lower())
                                 for level in LOG_LEVELS)
LOG_NAME_TO_LEVEL = OrderedDict((name, level)
                                for level, name in LOG_LEVEL_TO_NAMES.items())

VIDEO_EXTENSION = 'mp4'

def main(argv=None):
    args = parse_args(argv=argv)
    configure_logger(args)

    command = args.command
    if command == 'find':
        find_scenes(args)
    elif command == 'render':
        render_clips(args)
    elif command == 'listing':
        make_listing(args)
    else:
        raise RuntimeError('Invalid command {}'.format(args.command))


def parse_args(argv=None):
    if argv is None:
        argv = sys.argv
    parser = ArgumentParser()
    parser.add_argument('-l', '--log-level', choices=LOG_NAME_TO_LEVEL.keys(),
                        default=LOG_LEVEL_TO_NAMES[logging.INFO])
    subparsers = parser.add_subparsers(dest='command')
    find = subparsers.add_parser('find')
    find.add_argument('-t', '--threshold', default=10, type=int)
    find.add_argument('video_path')
    find.add_argument('output_dir')
    render = subparsers.add_parser('render')
    render.add_argument('-m', '--max-length', default=5, type=int)
    render.add_argument('-n', '--min-length', default=0.5, type=float)
    render.add_argument('scenes_path')
    render.add_argument('video_path')
    render.add_argument('output_dir')
    listing = subparsers.add_parser('listing')
    listing.add_argument('clips_dir')
    listing.add_argument('listing_path')
    return parser.parse_args(args=argv[1:])


def configure_logger(args):
    global logger
    logging.basicConfig(datefmt='%H:%M:%S',
                        format='[%(levelname).1s %(asctime)s] %(message)s',
                        level=LOG_NAME_TO_LEVEL[args.log_level])
    logger = logging.getLogger(__name__)


def find_scenes(args):
    video_path = args.video_path
    video_name = os.path.basename(video_path)
    video_stem, video_ext = os.path.splitext(video_name)

    output_dir = args.output_dir
    ensure_dir(output_dir)
    scenes_name = '{stem}.json'.format(stem=video_stem)
    scenes_path = os.path.join(output_dir, scenes_name)

    with video_capture(args.video_path) as cap:
        scene_splitter = SceneFinder(cap, args.threshold)
        scenes = scene_splitter.find_scenes()

    with open(scenes_path, 'w') as scenes_file:
        json.dump(scenes, scenes_file)


@contextmanager
def video_capture(vido_path):
    cap = cv2.VideoCapture(vido_path)
    yield cap
    cap.release()


class SceneFinder(object):
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
        logger.info('Scene: %.1f %.1f', *scene)
        self._scenes.append(scene)



def render_clips(args):
    video_path = args.video_path
    video_name = os.path.basename(video_path)
    video_stem, video_ext = os.path.splitext(video_name)

    output_dir = args.output_dir
    clips_dir = os.path.join(output_dir, video_stem)
    ensure_dir(output_dir)
    if os.path.isdir(clips_dir):
        shutil.rmtree(clips_dir)
    os.mkdir(clips_dir)

    with open(args.scenes_path) as scenes_file:
        scenes = json.load(scenes_file)

    def min_max_length(scene):
        return args.min_length < scene[1] - scene[0] < args.max_length
    scenes = filter(min_max_length, scenes)

    pool = multiprocessing.Pool()
    for index, (start_time, stop_time) in enumerate(scenes):
        clip_name = '{}-{}.{}'.format(video_stem, index, VIDEO_EXTENSION)
        clip_path = os.path.join(clips_dir, clip_name)
        if os.path.exists(clip_path):
            os.remove(clip_path)
        pool.apply_async(render_clip, [video_path, clip_path, start_time,
                                       stop_time])
    pool.close()
    pool.join()


def render_clip(video_path, clip_path, start_time, stop_time):
    logger.info('Rendering %s ...', clip_path)
    subprocess.check_call([
        '/usr/bin/ffmpeg',
        '-ss', str(start_time),
        '-t', str(stop_time - start_time),
        '-i', video_path,
        '-strict',
        '-2',
        clip_path,
    ])


def ensure_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def make_listing(args):
    listing = {'videos': []}
    for root, dirs, files in os.walk(args.clips_dir):
        for file_ in files:
            if os.path.splitext(file_)[1] != '.{}'.format(VIDEO_EXTENSION):
                continue
            common_prefix = os.path.commonprefix([args.clips_dir, root])
            path = os.path.join(root[len(common_prefix) + 1:], file_)
            listing['videos'].append(path)
    with open(args.listing_path, 'w') as listing_file:
        json.dump(listing, listing_file)


if __name__ == '__main__':
    sys.exit(main())

