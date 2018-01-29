import os
import sys
import pprint

import Tkinter
import Image, ImageTk


pp = pprint.PrettyPrinter(indent=4)

DATA_DIR = "C:\\Work\\research\\shadow_removal\\experiments\\training_images"

root = Tkinter.Tk()
sharp_shadows = []
current_image = ""

def button_click_exit_mainloop(event):
    if event.char == "q":
        root.destroy()
        quit()
        return
    elif event.char == "s":
        sharp_shadows.append(current_image)
    event.widget.quit() # this will cause mainloop to unblock.

root.bind("<Key>", button_click_exit_mainloop)
root.geometry('+%d+%d' % (100,100))
dirlist = [f for f in os.listdir(DATA_DIR) if f.endswith("_shad.png")]
old_label_image = None
for f in dirlist[100:500]:
    current_image = f
    try:
        image1 = Image.open(os.path.join(DATA_DIR, f))
        root.geometry('%dx%d' % (image1.size[0],image1.size[1]))
        tkpi = ImageTk.PhotoImage(image1)
        label_image = Tkinter.Label(root, image=tkpi)
        label_image.place(x=0,y=0,width=image1.size[0],height=image1.size[1])
        root.title(f)
        if old_label_image is not None:
            old_label_image.destroy()
        old_label_image = label_image
        root.mainloop() # wait until user clicks the window
    except Exception, e:
        # This is used to skip anything not an image.
        # Image.open will generate an exception if it cannot open a file.
        # Warning, this will hide other errors as well.
        pass

pp.pprint(sharp_shadows)

out = open("sharp_shadows.txt", "w")
for ss in sharp_shadows:
    out.write(ss + "\n")
out.close()

root.destroy()
quit()
