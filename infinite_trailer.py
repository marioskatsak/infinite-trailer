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

VIDEO_EXTENSION = 'webm'
YOUTUBE_VIDEO_FORMAT = '242'
YOUTUBE_AUDIO_FORMAT = '171'

THRESHOLD = 30

CLIPS_OUTPUT_DIR = os.path.join('html', 'clips')

MIN_CLIP_LENGTH = 1
MAX_CLIP_LENGTH = 5

LISTINGS_PATH = os.path.join('html', 'listings.json')

def main(argv=None):
    args = parse_args(argv=argv)
    configure_logger(args)

    command = args.command
    if command == 'bulk':
        bulk(args)
    elif command == 'download':
        download_trailer(args)
    elif command == 'find':
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

    bulk = subparsers.add_parser('bulk')
    bulk.add_argument('-m', '--max-length', default=MAX_CLIP_LENGTH, type=float)
    bulk.add_argument('-n', '--min-length', default=MIN_CLIP_LENGTH, type=float)
    bulk.add_argument('-c', '--trailers_config_path', default='trailers.json')
    bulk.add_argument('-l', '--listings_path', default=LISTINGS_PATH)
    bulk.add_argument('-o', '--trailers_output_dir', default='trailers')
    bulk.add_argument('-s', '--scenes_output_dir', default='scenes')
    bulk.add_argument('-t', '--clips_output_dir', default=CLIPS_OUTPUT_DIR)
    bulk.add_argument('-d', '--download', dest='download', action='store_true')
    bulk.add_argument('-D', '--skip-download', dest='download',
                      action='store_false')
    bulk.set_defaults(download=True)
    bulk.add_argument('-r', '--render', dest='render', action='store_true')
    bulk.add_argument('-R', '--skip-render', dest='render',
                      action='store_false')
    bulk.set_defaults(render=True)

    download = subparsers.add_parser('download')
    download.add_argument('youtube_id')
    download.add_argument('output_filename')
    download.add_argument('-v', '--video_format', default=YOUTUBE_VIDEO_FORMAT)
    download.add_argument('-a', '--audio_format', default=YOUTUBE_AUDIO_FORMAT)

    find = subparsers.add_parser('find')
    find.add_argument('-t', '--threshold', default=THRESHOLD, type=int)
    find.add_argument('video_path')
    find.add_argument('output_dir')

    render = subparsers.add_parser('render')
    render.add_argument('-m', '--max-length', default=MAX_CLIP_LENGTH,
                        type=float)
    render.add_argument('-n', '--min-length', default=MIN_CLIP_LENGTH,
                        type=float)
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


def bulk(args):
    with open(args.trailers_config_path) as trailers_config_file:
        trailers_config = json.load(trailers_config_file)

    trailers_output_dir = args.trailers_output_dir
    ensure_dir(trailers_output_dir)
    scenes_output_dir = args.scenes_output_dir
    ensure_dir(scenes_output_dir)
    clips_output_dir = args.clips_output_dir
    ensure_dir(clips_output_dir)

    # XXX: Only run task so OpenCV doesn't corrupt itself up, had problems when
    # opening another video in the same process, would open the video and
    # immediately close.
    pool = multiprocessing.Pool(maxtasksperchild=1)

    for trailer in trailers_config['trailers']:
        pool.apply_async(create_clips_for_trailer,
                        [trailer, trailers_output_dir, scenes_output_dir,
                         clips_output_dir, args.download])

    pool.close()
    pool.join()

    for trailer in trailers_config['trailers']:
        video_path = get_video_file_name(trailers_output_dir, trailer['name'])
        scene_file = get_scenes_file_name(video_path, scenes_output_dir)
        if args.render:
            _render_clips(video_path, clips_output_dir, scene_file,
                          min_length=args.min_length,
                          max_length=args.max_length)
    _make_listing(os.path.join(clips_output_dir, '..'))


def get_video_file_name(output_dir, name):
    return os.path.join(output_dir, name)


def create_clips_for_trailer(trailer, trailers_output_dir, scenes_output_dir,
                             clips_output_dir,
                             download=True):
    output_path = get_video_file_name(trailers_output_dir, trailer['name'])
    if download:
        _download_trailer(output_path, trailer['youtube_id'])
    logger.info('Searching %s', output_path)
    scenes_path = _find_scenes(output_path, scenes_output_dir)


def download_trailer(args):
    _download_trailer(args.output_filename,
                      args.youtube_id,
                      video_format=args.video_format,
                      audio_format=args.audio_format)

def _download_trailer(
        output_filename,
        youtube_id,
        video_format=YOUTUBE_VIDEO_FORMAT,
        audio_format=YOUTUBE_AUDIO_FORMAT):
    logger.info('Downloading %s ...', output_filename)
    subprocess.check_call([
        'youtube-dl',
        '-o', '{}'.format(output_filename),
        'https://www.youtube.com/watch?v={}'.format(youtube_id),
        '-f', '{}+{}'.format(video_format, audio_format)
    ])

    # XXX: youtube-dl leaves some artifacts of the audio and video streams it
    # downloaded so we'll delete them.
    def unlink_download_artifacts(output_filename, dl_format):
        extension = os.path.splitext(output_filename)[1]
        output_dir = os.path.dirname(os.path.realpath(output_filename))
        output_basename = os.path.basename(os.path.realpath(output_filename))
        basename = os.path.splitext(output_basename)[0]
        artifact = '{}.f{}{}'.format(basename, dl_format, extension)
        os.unlink(os.path.join(output_dir, artifact))

    unlink_download_artifacts(output_filename, video_format)
    unlink_download_artifacts(output_filename, audio_format)


def find_scenes(args):
    _find_scenes(args.video_path, args.output_dir, threshold=args.threshold)


def get_scenes_file_name(video_path, output_dir):
    video_name = os.path.basename(video_path)
    video_stem, video_ext = os.path.splitext(video_name)

    scenes_name = '{stem}.json'.format(stem=video_stem)
    return os.path.join(output_dir, scenes_name)

def _find_scenes(video_path, output_dir, threshold=THRESHOLD):
    ensure_dir(output_dir)
    scenes_path = get_scenes_file_name(video_path, output_dir)

    with video_capture(video_path) as cap:
        scene_splitter = SceneFinder(cap, threshold)
        scenes = scene_splitter.find_scenes()

    if len(scenes) == 0:
        logger.error('No scenes found for %s' % video_path)

    with open(scenes_path, 'w') as scenes_file:
        json.dump(scenes, scenes_file)

    return scenes_path


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
                logger.info('Stopping on frame %d' % self._start_index)
                if self._start_index == 0:
                    logger.error('Not able to read any frames')
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
    _render_clips(
        args.video_path,
        args.output_dir,
        args.scenes_path,
        min_length=args.min_length,
        max_length=args.max_length
    )


def _render_clips(video_path, output_dir, scenes_path,
                  min_length=MIN_CLIP_LENGTH, max_length=MAX_CLIP_LENGTH):
    video_name = os.path.basename(video_path)
    video_stem, video_ext = os.path.splitext(video_name)

    clips_dir = os.path.join(output_dir, video_stem)
    ensure_dir(output_dir)
    if os.path.isdir(clips_dir):
        shutil.rmtree(clips_dir)
    os.mkdir(clips_dir)

    with open(scenes_path) as scenes_file:
        scenes = json.load(scenes_file)

    def min_max_length(scene):
        return min_length < scene[1] - scene[0] < max_length
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
        '-c:v', 'libvpx',
        '-c:a', 'libvorbis',
        clip_path,
    ])


def ensure_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def make_listing(args):
    _make_listing(args.clips_dir, listing_path=args.listing_path)


def _make_listing(clips_dir, listing_path=LISTINGS_PATH):
    listing = {'videos': []}
    for root, dirs, files in os.walk(clips_dir):
        for file_ in files:
            if os.path.splitext(file_)[1] != '.{}'.format(VIDEO_EXTENSION):
                continue
            common_prefix = os.path.commonprefix([clips_dir, root])
            path = os.path.join(root[len(common_prefix) + 1:], file_)
            listing['videos'].append(path)
    with open(listing_path, 'w') as listing_file:
        json.dump(listing, listing_file)


if __name__ == '__main__':
    sys.exit(main())

