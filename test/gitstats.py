#!/usr/bin/python

import subprocess, time
from math import *
from numpy import *
from matplotlib import pyplot
import main

#----( git interface )--------------------------------------------------------

def return_command (command):
  process = subprocess.Popen(command, shell = True, stdout = subprocess.PIPE)
  return process.communicate()[0]

def git (*args):
  return return_command(' '.join(['git'] + [str(arg) for arg in args]))

def get_timestamps ():
  #print 'getting timestamps from git log'
  log = git("log --pretty=format:'%h %at'").strip().split('\n')
  log.reverse()
  return [(hash,int(time)) for (hash,time) in map(str.split, log)]

def get_stats (hash):
  #print 'getting stats from git diff'
  'returns: # files changed, # insertions(+), # deletions(-)'
  line = git('diff %s~ %s --shortstat' % (hash,hash)).strip()
  if line:
    print(line)
    parts = [part.strip().split(' ', 1) for part in line.split(',')]
    parts = {key.strip(): int(val) for (val, key) in parts}
    return (parts.get('files changed', parts.get('file changed', 0)),
            parts.get('insertions(+)', 0),
            parts.get('deletions()', 0))
  else:
    return (0,0,0)

#----( statistics )-----------------------------------------------------------

def fourier_kde (data, weights = None, acuity = None, size = None):
  'fourier kernel density estimation'

  if weights is None: weights = ones(len(data))
  if acuity is None: acuity = sqrt(len(data))
  if size is None: size = round(10 * acuity)

  acuity = float(acuity)
  size = int(size)
  dim = int(round(3 * acuity))

  print('Fourier KDE acuity = %g, dim = %g' % (acuity, dim))

  n = array(list(range(dim)))
  omega = 2.0j * pi * n
  blur = exp(-pi * (n/acuity)**2)

  coeff = zeros(dim, dtype=complex)
  for point,weight in zip(data,weights):
    coeff += weight * exp(omega * point)
  coeff *= blur
  coeff[1:] *= 2 # since we ignore negative frequencies

  t = ((0.5 + array(list(range(size)))) / size)
  basis = exp(omega.conj().reshape((1,dim)) * t.reshape((size,1)))

  return t,dot(basis, coeff).real

#----( plotting )-------------------------------------------------------------

@main.command
def timeline ():
  'Plots timeline of insertions & deletions'

  timestamps = get_timestamps()
  stats = [get_stats(hash) for (hash,_) in timestamps]

  time = [t for (hash, t) in timestamps]
  diff = cumsum([a+d for (f,a,d) in stats])
  total = cumsum([a-d for (f,a,d) in stats])

  pyplot.figure()
  pyplot.plot(time, total, 'k-', label = 'total lines')
  pyplot.plot(time, diff, 'r-', label = 'diff')
  pyplot.xlabel('Time (ms?)')
  pyplot.ylabel('Lines')
  pyplot.title('Timeline of %i commits' % len(timestamps))

  pyplot.savefig('timeline.png')
  pyplot.show()

@main.command
def progress ():
  'Plots progress by insertions & deletions'

  timestamps = get_timestamps()
  stats = [get_stats(hash) for (hash,_) in timestamps]

  diff = cumsum([a+d for (f,a,d) in stats])
  total = cumsum([a-d for (f,a,d) in stats])

  pyplot.figure()
  pyplot.plot(diff, total, 'k.')
  pyplot.xlabel('Total change')
  pyplot.ylabel('Total size')
  pyplot.title('Progress of %i commits' % len(timestamps))

  pyplot.savefig('progress.png')
  pyplot.show()

@main.command
def commitload ():
  'Draws commit histogram of hour,day'

  timestamps = get_timestamps()
  #stats = [get_stats(hash) for (hash,_) in timestamps]

  def to_hour (t): return t.tm_hour + (t.tm_min + t.tm_sec / 60.0) / 60.0
  def to_wday (t): return t.tm_wday + to_hour(t) / 24.0

  secs = [t for hash,t in timestamps]
  total_secs = max(secs) - min(secs)
  total_hours = total_secs / 3600.0
  total_days = total_hours / 24.0
  total_weeks = total_days / 7.0

  times = [time.localtime(t) for hash,t in timestamps]
  hours = array(list(map(to_hour, times)))
  wdays = array(list(map(to_wday, times)))

  hour,commits_hour = fourier_kde(hours / 24.0, acuity = 12)
  wday,commits_wday = fourier_kde(wdays / 7.0, acuity = 14)

  hour *= 24.0;   commits_hour /= total_hours
  wday *= 7.0;    commits_wday /= total_days

  def zero_ylim (ax):
    y0,y1 = ax.get_ylim()
    ax.set_ylim((0,y1))

  fig = pyplot.figure()

  ax = fig.add_subplot(2,1,1)
  ax.plot(hour, commits_hour)
  ax.set_xlabel('hour of day')
  ax.set_ylabel('commits / hour')
  ax.set_xlim((0,24))
  zero_ylim(ax)

  ax.set_title('Distribution of %i commits over %i days'
      % (len(times), int(total_days)))

  ax = fig.add_subplot(2,1,2)
  ax.plot(wday, commits_wday)
  ax.set_xlabel('day of week')
  ax.set_ylabel('commits / day')
  ax.set_xlim((0,7))
  zero_ylim(ax)

  pyplot.savefig('commitload.png')
  pyplot.show()

@main.command
def changeload ():
  'Draws change histogram of hour,day'

  timestamps = get_timestamps()
  stats = [get_stats(hash) for (hash,_) in timestamps]

  secs = [t for hash,t in timestamps]
  total_secs = max(secs) - min(secs)
  total_hours = total_secs / 3600.0
  total_days = total_hours / 24.0
  total_weeks = total_days / 7.0

  total_weeks *= 1.02 # zero-pad by 2% to prevent cycling in kde

  epoch = array([float(t) for hash,t in timestamps])
  times = [time.localtime(t) for hash,t in timestamps]

  oldest,newest = epoch[0],epoch[-1]
  age_01 = (epoch - newest) / (oldest - newest)

  def to_week (t): return (t - epoch[0]) / (3600.0 * 24.0 * 7.0)
  def to_hour (t): return t.tm_hour + (t.tm_min + t.tm_sec / 60.0) / 60.0
  def to_wday (t): return t.tm_wday + to_hour(t) / 24.0

  weeks = array(list(map(to_week, epoch)))
  wdays = array(list(map(to_wday, times)))
  hours = array(list(map(to_hour, times)))

  changes = array([a+d/4 for (_,a,d) in stats]) # deletions are 1/4 as heavy

  week,changes_week = fourier_kde(weeks / total_weeks, changes, acuity = 64)
  wday,changes_wday = fourier_kde(wdays / 7.0, changes, acuity = 14)
  hour,changes_hour = fourier_kde(hours / 24.0, changes, acuity = 12)

  week *= total_weeks;  changes_week /= total_weeks
  hour *= 24.0;         changes_hour /= total_hours
  wday *= 7.0;          changes_wday /= total_days

  def cmap (a): return (1 - 0.75 * a, 0.25 * a, 0.25 * a)
  def zero_ylim (ax):
    y0,y1 = ax.get_ylim()
    ax.set_ylim((0,y1))

  fig = pyplot.figure()

  # entire history
  ax = fig.add_subplot(3,1,1)
  scatter = ax.twinx()
  for t,c,a in zip(weeks, changes, age_01):
    scatter.plot([t], [c], 'r.', color=cmap(a))
  scatter.set_ylabel('commit size')
  scatter.set_yscale('log')

  ax.plot(week, changes_week, 'k-')
  ax.set_xlabel('week after first commit')
  ax.set_ylabel('changes / week')
  ax.set_xlim((0,total_weeks))
  zero_ylim(ax)

  # title
  ax.set_title('Distribution of %i changes in %i commits over %i days'
      % (sum(changes), len(changes), int(total_days)))

  # typical week
  ax = fig.add_subplot(3,1,2)
  scatter = ax.twinx()
  for t,c,a in zip(wdays, changes, age_01):
    scatter.plot([t], [c], 'r.', color=cmap(a))
  scatter.set_ylabel('commit size')
  scatter.set_yscale('log')

  ax.plot(wday, changes_wday, 'k-')
  ax.set_xlabel('day of typical week')
  ax.set_ylabel('changes / day')
  ax.set_xlim((0,7))
  zero_ylim(ax)

  # typical day
  ax = fig.add_subplot(3,1,3)
  scatter = ax.twinx()
  for t,c,a in zip(hours, changes, age_01):
    scatter.plot([t], [c], 'r.', color=cmap(a))
  scatter.set_ylabel('commit size')
  scatter.set_yscale('log')

  ax.plot(hour, changes_hour, 'k-')
  ax.set_xlabel('hour of typical day')
  ax.set_ylabel('changes / hour')
  ax.set_xlim((0,24))
  zero_ylim(ax)

  pyplot.savefig('changeload.png')
  pyplot.show()

@main.command
def todo (max_items = 32):
  'prints a todo list'

  lines = git('grep', '-n', 'TO'+'DO', '..')

  tuples = [line.split(':',2) for line in lines.split('\n') if line]

  dated = []
  for t in tuples:
    fname,lineno,text = t
    date = git('blame', '--porcelain', '-L %s,%s' % (lineno,lineno), fname,
        '|', 'grep', 'author-time').split()[1]
    dated.append((
      int(date),
      fname,
      lineno,
      text.split('TO'+'DO')[-1].strip()))

  dated.sort()
  dated.reverse()
  dated = dated[:max_items]
  for date,fname,lineno,text in dated:
    #print text
    print('\033[1m%s\033[0m\n(%s:%s)' % (text, fname, lineno))

if __name__ == '__main__': main.main()

