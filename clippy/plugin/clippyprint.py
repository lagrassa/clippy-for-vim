from __future__ import print_function
import vim                                                              


def welcome_clippy():
    welcome_speech_bubble=" __________________\n/                 \ \n|Guess who is back!|\n|It's your friend, |\n|Clippy!!!         |\n\_______________  _/\n                \/\n"
    small_clippy = "   __\n  /  \ \n  |  |\n  O  O\n  || ||\n  || ||\n  |\_/|\n  \___/  \n"
    print(welcome_speech_bubble)
    print(small_clippy)

def print_bubble(text, widthBox):                                   
    text = text.replace("\n"," ")
    text = text.replace("\r"," ") 
    quote_left = text[:]
    top_bubble ="  " + "".join(["_"]*(widthBox-3))+"  "
    top_bubble_with_lines ="/" + "".join([" "]*(widthBox-3))+"\ "
    print(top_bubble,end='')
    print(top_bubble_with_lines,end='')
    while True:
        if len(quote_left) < widthBox-3:
            spaces = "".join([" "]*(widthBox-len(quote_left)-3))
            print("|"+quote_left+spaces+"|",end='')
            break;
        else:
           quote_to_print = quote_left[:widthBox-4]
           print("|"+quote_to_print+"|\n",end='')
           quote_left = quote_left[widthBox-4:]
    big_clippy = "               .~=777?                  \n               :I    .7.                \n              :I.     D?                \n           O  : Z     $?                \n           ,.,~      .MNNN              \n         ,..ZNM.~    ,,~  .             \n         ~.NDNM.~  ...8M.,.             \n          +:,,:=I  ,.NNNN.=             \n             OO.    :,..,=?             \n             $$      88 .               \n             Z$$$    OZ .$              \n             Z7$7    O7 IO              \n             ZII7    O= ?.              \n             Z+.?    O= +.              \n             =~ ?    7? ?               \n              +.$=   +? I?              \n              $, 7~~:+  Z7              \n              8:    ..  OI              \n              .+        7?              \n              .8:       ?               \n                8~    =~.               \n                  OI?I? .  \n"
    bottom_bubble = " \\"+"".join(["_"]*(widthBox-6))+"  "+"_/"
    tip_of_bottom_bubble = "".join([" "]*(widthBox-4))+"\/"
    print(bottom_bubble,end="")
    print(tip_of_bottom_bubble)
    print(big_clippy)

def clippy_shell(scriptname_with_args, widthBox):
    quote = "Let me execute the command:" + scriptname_with_args+ " for you!  "
    print_bubble(quote, widthBox)

def clippy_friendly_output(output,widthBox):
    output_quote = "I found the answer for you! " + output
    if "error" in output:
        output_quote += "   I'm sorry about that....Better luck next time!!"
    output_quote += "Was that helpful?"
    print_bubble(output_quote,widthBox)


