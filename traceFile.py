#import windowManager3D as wm
from miscUtil import timeString
import local
import planGlobals as glob
import os
import pdb

# only import if we need it
wm = None
def importWm():
    global wm
    if not wm:
        import windowManager3D
        wm = windowManager3D
    return wm

# debugOn  : print to tty
# pauseOn : pause tty
# logOn : print to log

# tag lists contain: *, symbol, (symbol, *), or (symbol, symbol)
# tag in a tr statement can be: symbol or (symbol, symbol)

# Return true if this tag should be traced, given a particular list of tags
def traced(genTag, tags, skip = False, keys = {}):
    return (not skip) and \
      ((not glob.inHeuristic) or ('debugInHeuristic' in tags) or
       keys.get('h', False)) and \
      (genTag == '*'  or  ('*' in tags) or
                         (genTag in tags) or ((genTag[0], '*') in tags))

# Always print and log this.  
def trAlways(*msg, **keys):
    keys['h']=True                        # do this in the heuristic as well
    tr('*', *msg, **keys)

# Decide whether to write into log
def log(tag, **keys):
    return traced(tag, glob.logOn, False, keys)

# Decide whether to write to tty
def debug(tag, **keys):
    return traced(tag, glob.debugOn, False, keys)

# Decide whether to pause tty.  Default to no, even if '*'
def pause(tag, **keys):
    return keys.get('pause', False) or \
      ((tag != '*') and traced(tag, glob.pauseOn, False, keys))

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
#    h: do it in the heuristic 


def tr(genTag, *msg, **keys):
    if keys.get('skip', False) or 'noTrace' in glob.debugOn:  return
        
    if glob.inHeuristic and htmlFileH:
        targetFile = htmlFileH
    elif (not glob.inHeuristic) and htmlFile:
        targetFile = htmlFile
    else:
        targetFile = None

    doLog = log(genTag, **keys)
    doDebug = debug(genTag, **keys)
    doPause = pause(genTag, **keys)
    ol = keys.get('ol', True)
    tagStr = '' if genTag == '*' else genTag+': '

    # Logging text
    if doLog and msg and targetFile:
        targetFile.write('<pre>'+tagStr)
        terminator = ' ' if ol else '\n    '
        for m in msg:
            mo = (str(m)+terminator).replace('\n', '<br>')
            mo = mo.replace('\\n', '<br>')
            targetFile.write(mo)
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
            importWm()
            wm.getWindow(w).update()
        raw_input(tagStr+' go?')

pngFileId = 0
htmlFile = None
htmlFileH = None
htmlFileId = 0
dirName = None

def traceHeader(h):
    for targetFile in (htmlFile, htmlFileH):
        if targetFile:
            targetFile.write('<h1>'+h+'</h1>')

def traceStart(name='log'):
    global htmlFile, htmlFileH, htmlFileId, dirName
    dirName = local.genDir + name + '_'+timeString()
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
    if htmlFile:
        htmlFile.write(header)
    if htmlFileH:
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
    wm = importWm()
    global pngFileId
    for win in windows:
        pngFileName = local.pngGen%(dirName, str(pngFileId))
        if wm.getWindow(win).window.modified:
            wm.getWindow(win).window.saveImage(pngFileName)
            pngFileId += 1
            if targetFile:
                targetFile.write('<br><img src="%s" border=1 alt="%s"><br>\n'\
                                 %(pngFileName,win))
        else:
            if targetFile:
                targetFile.write('<br><b>No change to window %s</b><br>\n'%win)

def trLog(string):
    print string
    if htmlFile:
        htmlFile.write('<pre>%s</pre>\n'%string)