if !has('python')
    finish
endif

" Vim comments start with a double quote.
" Function definition is VimL. We can mix VimL and Python in
" function definition.

function! ShowClippy()
python << endpython
import sys
sys.path.append(".")
import clippyprint as cp
import clippyprint as cp
cp.welcome_clippy()
endpython
endfunction

function InsultMe()
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

function! ExecuteScript(scriptname_with_args)
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
cp.clippy_friendly_output(var,width)
endpython
endfunction

nmap insult ;s :call InsultMe()

