from __future__ import print_function
import vim                                                              

small_clippy = "   __\n  /  \ \n  |  |\n  O  O\n  || ||\n  || ||\n  |\_/|\n  \___/  \n"

big_clippy = "               .~=777?                  \n               :I    .7.                \n              :I.     D?                \n           O  : Z     $?                \n           ,.,~      .MNNN              \n         ,..ZNM.~    ,,~  .             \n         ~.NDNM.~  ...8M.,.             \n          +:,,:=I  ,.NNNN.=             \n             OO.    :,..,=?             \n             $$      88 .               \n             Z$$$    OZ .$              \n             Z7$7    O7 IO              \n             ZII7    O= ?.              \n             Z+.?    O= +.              \n             =~ ?    7? ?               \n              +.$=   +? I?              \n              $, 7~~:+  Z7              \n              8:    ..  OI              \n              .+        7?              \n              .8:       ?               \n                8~    =~.               \n                  OI?I? .  \n"

def welcome_clippy(width):
    quote = "Welcome back! It's your friend Clippy! Use 'i' to enter insert mode to write text, '<ESC>' to enter normal mode. For more help, try to :call ClippyHelp(<topic>) where topic = {navigation, copypaste, repeatedcommands, deletion, or search}"
    print_bubble(quote, text)

def help_menu(command, width):
    if command == "navigation":
        help_message = "k = up, j = down, l = right, h = left 0 = beginning of line, $ = end of line, b = beginning of word, e = end of word"
    elif command == "copypaste":
        help_message = "Use 'v' to enter visual mode to select the text. Deleted text is automatically added to the paste buffer. 'y' will add text to the paste buffer without deletion. 'p' will paste "
    elif command == "repeatedcommands":
        help_message = "Add a number before a command to repeat it. For instance, deleting 8 lines is '8dd'"
    elif command == "deletion":
        help_message = "'dd' deletes a line, 'd' deletes selected text. 'd' before a location qualifier (like 'b', or 'e') deletes from the cursor to that location"
    elif command == "search":
        help_message = "Search for terms with '/<search term>'. 'n' brings you to the next instance, and '<Shift> + n' brings the cursors to the previous instance"
    else:
        raise ValueError("Unsupported help command")
    print_bubble(help_message, width, size = 'small')

def print_bubble(text, widthBox, size='large'):                      
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
    bottom_bubble = " \\"+"".join(["_"]*(widthBox-6))+"  "+"_/"
    tip_of_bottom_bubble = "".join([" "]*(widthBox-4))+"\/"
    print(bottom_bubble,end="")
    print(tip_of_bottom_bubble)
    if size == 'large':
        print(big_clippy)
    else:
        print(small_clippy)

def clippy_shell(scriptname_with_args, widthBox):
    quote = "Let me execute the command:" + scriptname_with_args+ " for you!  "
    print_bubble(quote, widthBox, size='small')

def clippy_friendly_output(output,widthBox):
    output_quote = "I found the answer for you! " + output
    if "error" in output:
        output_quote += "   I'm sorry about that....Better luck next time!!"
    output_quote += "Was that helpful?"
    print_bubble(output_quote,widthBox)

def insult(width):
    insult_list = ["Looks like you need a template to save you some time", "Good luck reinventing the wheel", "q! q! q! Do it! It's not like you wrote anything of value anyway", "It looks like you are angrily mashing the keyboard, do you want me to turn on caps lock?","It looks like you're having a problem deleting this useless file. Would you like help with that?" ]
    import random
    i = random.randint(0,len(insult_list))
    print_bubble(insult_list[i],width)
