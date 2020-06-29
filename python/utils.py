
# frequently used unit convertion
au2eV = 27.211386245988
au2A = 0.529177210903

def printtbox(arg):
    """
    This function is a print decorated with a few characters so that the
    print function present a characteristic string. Useful for postprocess.
    """
    print("--> Toolkit: {0}".format(arg))

def file_content(file, num):
    # read a specific line of file or return the block
    # file: enter file name
    # num: a integer -> return specific line content
    #      a tuple (num1, num2) -> return the line content
    #                              between num1 and num2-1
    #      a tuple (num1, ) -> return the line content from
    #                          num1, to the end of file
    if isinstance(num, int):
        with open(file) as f:
            for _idx, line in enumerate(f):
                if _idx == num:
                    return line
    elif isinstance(num, tuple):
        content = ""
        if len(num) == 2:
            with open(file) as f:
                for _idx, line in enumerate(f):
                    if (_idx >= num[0]) and (_idx < num[1]):
                        content += line
                    elif _idx >= num[1]:
                        break
                    else:
                        continue
            return content
        elif len(num) == 1:
            with open(file) as f:
                for _idx, line in enumerate(f):
                    if (_idx >= num[0]) :
                        content += line
            return content
        else:
            raise ValueError("The length of range is wrong!")
