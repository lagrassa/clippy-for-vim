import windowManager3D as wm
from miscUtil import timeString
import local
import planGlobals as glob
import os

# debugOn  : print to tty
# pauseOn : pause tty
# logOn : print to log

# tag lists contain: *, symbol, (symbol, *), or (symbol, symbol)
# tag in a tr statement can be: symbol or (symbol, symbol)

# Return true if this tag should be traced, given a particular list of tags
def traced(genTag, tags, skip = False):
    return (not skip) and \
      ((not glob.inHeuristic) or ('debugInHeuristic' in tags)) and \
      (genTag == '*'  or  ('*' in tags) or
            (genTag in tags) or ((genTag[0], '*') in tags))

# Always print and log this.  
def trAlways(*msg, **keys):
    tr('*', *msg, **keys)

# Decide whether to write into log
def log(tag, skip = False):
    return traced(tag, glob.logOn, skip)

# Decide whether to write to tty
def debug(tag, skip = False):
    return traced(tag, glob.debugOn, skip)

# Decide whether to pause tty.  Default to no, even if '*'
def pause(tag, skip=False):
    return (tag != '*') and traced(tag, glob.pauseOn, skip)

def debugMsg(tag, *msgs):
    if debug(tag):
        print tag, ':'
        for m in msgs:
            print '    ', m
    if pause(tag):
        raw_input(tag+'-Go?')

def debugMsgSkip(tag, skip = False, *msgs):
    if debug(tag, skip):
        print tag, ':'
        for m in msgs:
            print '    ', m
    if pause(tag, skip):
        raw_input(tag+'-Go?')

def debugDraw(tag, obj, window, color = None, skip=False):
    if debug(tag, skip):
        obj.draw(window, color = color)
    if pause(tag, skip):
        raw_input(tag+'-Go?')

# keys is a dictionary
# Possible keywords:  draw, snap, pause
#    draw: draws into a window
#    snap: puts image of listed windows into the log
#    pause: pauses
#    ol: write onto single line, if true
#    skip: skip this whole thing


def tr(genTag, *msg, **keys):
    if keys.get('skip', False):  return
        
    if glob.inHeuristic and htmlFileH:
        targetFile = htmlFileH
    elif (not glob.inHeuristic) and htmlFile:
        targetFile = htmlFile
    else:
        targetFile = None

    doLog = log(genTag)
    doDebug = debug(genTag)
    doPause = keys['pause'] if ('pause' in keys) else pause(genTag)
    ol = keys.get('ol', True)
    tagStr = '' if genTag == '*' else genTag

    # Logging text
    if doLog and msg and targetFile:
        targetFile.write('<pre>'+' '+\
                         '(%d)%s'%(glob.planNum, tagStr)+\
                          ': '+str(msg[0])+'</pre>\n')
        terminator = ' ' if ol else '\n    '
        targetFile.write('<pre>')
        for m in msg[1:]:
            targetFile.write(str(m)+terminator)
        targetFile.write('</pre>\n')

    # Drawing
    if doLog or doDebug:
        draw = keys.get('draw', [])
        for obj in draw:
            if isinstance(obj, (list, tuple)):
                obj[0].draw(*obj[1:])
            else:
                obj.draw('W')
        if doLog and keys.get('snap', []):
            snap(targetFile, *keys['snap'])

    # Printing to console
    if doDebug and msg:
        print tagStr,
        if ol:
            for m in msg: print m,
            print '\n',
        else:
            for m in msg:
                print '    ', m

    # Pausing
    if doPause:
        windows = keys.get('snap', [])
        for w in windows:
            wm.getWindow(w).update()
        raw_input(tagStr+' go?')

pngFileId = 0
htmlFile = None
htmlFileH = None
htmlFileId = 0
dirName = None

def traceStart():
    global htmlFile, htmlFileH, htmlFileId, dirName
    dirName = local.genDir + 'log_'+timeString()
    if not os.path.exists(dirName):
        os.makedirs(dirName)
    htmlFile = open(local.htmlGen%(dirName, str(htmlFileId)), 'w')
    htmlFileH = open(local.htmlGenH%(dirName, str(htmlFileId)), 'w')
    htmlFileId += 1
    # Could use this for css styles
    header = '''
<!DOCTYPE html>
<html>
<body>
'''
    htmlFile.write(header)
    htmlFileH.write(header)

def traceEnd():
    global htmlFile, htmlFileH
    footer = '''
</body>'''
    if htmlFile: 
        htmlFile.write(footer)
        htmlFile.close()
        htmlFile = None
    if htmlFileH:
        htmlFileH.write(footer)
        htmlFileH.close()
        htmlFileH = None

def snap(targetFile, *windows):
    global pngFileId
    for win in windows:
        pngFileName = local.pngGen%(dirName, str(pngFileId))
        if wm.getWindow(win).window.modified:
            wm.getWindow(win).window.saveImage(pngFileName)
            pngFileId += 1
            targetFile.write('<br><img src="%s" border=1 alt="%s"><br>\n'%(pngFileName,win))
        else:
            targetFile.write('<br><b>No change to window %s</b><br>\n'%win)

def trLog(string):
    print string
    if htmlFile:
        htmlFile.write('<pre>%s</pre>\n'%string)
