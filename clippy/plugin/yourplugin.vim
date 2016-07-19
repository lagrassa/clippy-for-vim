if !has('python')
    finish
endif

" Vim comments start with a double quote.
" Function definition is VimL. We can mix VimL and Python in
" function definition.

function! ShowClippy()
    python << endpython
    import vim
    welcome_speech_bubble=" __________________\n/                 \ \n|Guess who is back!|\n|It's your friend, |\n|Clippy!!!         |\n\_______________  _/\n                \/\n"
    small_clippy = "   __\n  /  \ \n  |  |\n  O  O\n  || ||\n  || ||\n  |\_/|\n  \___/  \n"
    print(welcome_speech_bubble)
    print(small_clippy)
endpython
endfunction



function! BigClippy()
python << endpython
# the vim module contains everything we need to interface with vim from
# python. We need urllib2 for the web service consumer.
import vim
big_clippy = "               .~=777?                  \n               :I    .7.                \n              :I.     D?                \n           O  : Z     $?                \n           ,.,~      .MNNN              \n         ,..ZNM.~    ,,~  .             \n         ~.NDNM.~  ...8M.,.             \n          +:,,:=I  ,.NNNN.=             \n             OO.    :,..,=?             \n             $$      88 .               \n             Z$$$    OZ .$              \n             Z7$7    O7 IO              \n             ZII7    O= ?.              \n             Z+.?    O= +.              \n             =~ ?    7? ?               \n              +.$=   +? I?              \n              $, 7~~:+  Z7              \n              8:    ..  OI              \n              .+        7?              \n              .8:       ?               \n                8~    =~.               \n                  OI?I? .  \n"
print(big_clippy) 
endpython
endfunction

function! ExecuteScript(scriptname_with_args)
python << endpython
import vim
scriptname_with_args = vim.eval("a:scriptname_with_args")
command_name = scriptname_with_args
quote = "Do you want me to execute the command: \n" + command_name+ "for you?"
vim.command("let variable=system('!python printthing.py')")
var = vim.eval("variable")
#output = vim.command(scriptname_with_args)
output_quote = "I found the answer for you! \n" + var + "\n Was that helpful?"
print(output_quote)
endpython
endfunction

