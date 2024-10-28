#!/bin/bash
isort --skip data --skip work_dirs .
black --exclude '/data/' --exclude '/work_dirs/' . --line-length 100
