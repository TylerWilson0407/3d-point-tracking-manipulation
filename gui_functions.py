import tkinter as tk

class Confirmation:
    def __init__(self):
        self.root = tk.Tk()
        self.confirmed = None

    def confirm(self, true_false):
        self.confirmed = true_false
        self.root.destroy()

    def button_accept_cancel(self):
        frame = tk.Frame(self.root)
        frame.pack()

        slogan = tk.Button(frame,
                           text="Accept",
                           fg="green",
                           command=lambda: self.confirm(True))
        slogan.pack(side=tk.LEFT)
        button = tk.Button(frame,
                           text="Cancel",
                           fg="red",
                           command=lambda: self.confirm(False))
        button.pack(side=tk.LEFT)

        self.root.mainloop()


def button_confirm():
    accept = Confirmation()
    accept.button_accept_cancel()
    return accept.confirmed
