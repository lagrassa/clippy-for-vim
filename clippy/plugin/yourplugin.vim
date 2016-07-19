if !has('python')
    finish
endif

" Vim comments start with a double quote.
" Function definition is VimL. We can mix VimL and Python in
" function definition.

function! ShowClippy()
python << endpython
import vim
import clippyprint as cp
cp.welcome_clippy()
endpython
endfunction



function! ExecuteScript(scriptname_with_args)
python << endpython
from __future__ import print_function
import vim
import sys
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

