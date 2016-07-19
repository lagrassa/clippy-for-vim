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
from __future__ import print_function
import vim
import pdb
scriptname_with_args = vim.eval("a:scriptname_with_args")
command_name = scriptname_with_args
width = int(vim.eval("winwidth(0)"))
def print_bubble(text):
    quote_left = text[:]
    top_bubble ="  " + "".join(["_"]*(width-3))+"  " 
    top_bubble_with_lines ="/" + "".join([" "]*(width-3))+"\ "
    print(top_bubble,end='')
    print(top_bubble_with_lines,end='')
    while True:
        if len(quote_left) < width:
            spaces = "".join([" "]*(width-len(quote_left)-3))
            print("|"+quote_left+spaces+"|",end='')
            break;
        else:
           quote_to_print = quote_left[:width-6]
           print("|"+quote_to_print+"|"+"\n",end='')
           quote_left = quote_left[width-6:]
       


    big_clippy = "               .~=777?                  \n               :I    .7.                \n              :I.     D?                \n           O  : Z     $?                \n           ,.,~      .MNNN              \n         ,..ZNM.~    ,,~  .             \n         ~.NDNM.~  ...8M.,.             \n          +:,,:=I  ,.NNNN.=             \n             OO.    :,..,=?             \n             $$      88 .               \n             Z$$$    OZ .$              \n             Z7$7    O7 IO              \n             ZII7    O= ?.              \n             Z+.?    O= +.              \n             =~ ?    7? ?               \n              +.$=   +? I?              \n              $, 7~~:+  Z7              \n              8:    ..  OI              \n              .+        7?              \n              .8:       ?               \n                8~    =~.               \n                  OI?I? .  \n"
    bottom_bubble = " \\"+"".join(["_"]*(width-6))+"  "+"_/"
    tip_of_bottom_bubble = "".join([" "]*(width-4))+"\/"
    print(bottom_bubble,end="")
    print(tip_of_bottom_bubble)
    print(big_clippy)
quote = "Let me execute the command:" + command_name+ " for you!  "
print_bubble(quote)

vim.command("let variable=system('"+scriptname_with_args+"')")
var = vim.eval("variable")
#output = vim.command(scriptname_with_args)
var.rstrip()
output_quote = "I found the answer for you! " + var + " Was that helpful?"
print_bubble(output_quote)
endpython
endfunction

