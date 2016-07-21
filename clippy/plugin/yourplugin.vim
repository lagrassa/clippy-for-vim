if !has('python')
    finish
endif


function! ClippyReact(command)
python << endpython
import sys
import vim
sys.path.append(".")
import clippyprint as cp
width = int(vim.eval("winwidth(0)"))
command = vim.eval("a:command")
cp.react(command,width)
endpython
endfunction

function! ShowClippy()
python << endpython
import sys
import vim
sys.path.append(".")
import clippyprint as cp
width = int(vim.eval("winwidth(0)"))
cp.welcome_clippy(width)
endpython
endfunction

function! InsultMe()
python << endpython
import sys
import vim
sys.path.append(".")
import clippyprint as cp
import clippyprint as cp
width = int(vim.eval("winwidth(0)"))
cp.insult(width)
endpython
endfunction

function! ClippyHelp(help_topic)
python << endpython
import sys
import vim
sys.path.append(".")
import clippyprint as cp
help_topic = vim.eval("a:help_topic")
width = int(vim.eval("winwidth(0)"))
cp.help_menu(help_topic, width)
endpython
endfunction


function! ClippyExecute(scriptname_with_args)
python << endpython
import sys
import vim
sys.path.append(".")
import clippyprint as cp

scriptname_with_args = vim.eval("a:scriptname_with_args")
width = int(vim.eval("winwidth(0)"))

cp.clippy_shell(scriptname_with_args,width)

vim.command("let variable=system('"+scriptname_with_args+"')")
var = vim.eval("variable")
cp.clippy_friendly_output(var,width, scriptname_with_args)
endpython
endfunction

nmap insult ;s :call InsultMe()

let small_clippy = "  __\n  /  \\ \n  |  |\n  O  O\n  || ||\n  || ||\n  |\\_/|\n  \\___/  \n"
let insert_message = "It looks like you're trying to leave insert mode. You are now in normal mode. Type : to enter commands \n "
let cursor_message = "Tip: Moving the cursor is much more efficient in normal mode! call ClippyHelp('navigation') to learn how."
autocmd InsertLeave * :echo insert_message . small_clippy
autocmd CursorMovedI * :echo cursor_message . small_clippy 
autocmd BufWinLeave * :echo "Where are you going?" . small_clippy

