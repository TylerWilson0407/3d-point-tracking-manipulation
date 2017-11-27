import tkinter as tk

class ButtonConfirm:
    def __init__(self):
        self.root = tk.Tk()
        self.confirmed = None

    def confirm(self, true_false):
        self.confirmed = true_false
        self.root.destroy()

    def button_accept_cancel(self):
        frame = tk.Frame(self.root)
        frame.pack()

        button_confirm = tk.Button(frame,
                           text="Accept",
                           fg="green",
                           command=lambda: self.confirm(True))
        button_confirm.pack(side=tk.LEFT)
        button_deny = tk.Button(frame,
                           text="Deny",
                           fg="red",
                           command=lambda: self.confirm(False))
        button_deny.pack(side=tk.LEFT)

        self.root.mainloop()


def button_confirm():
    accept = ButtonConfirm()
    accept.button_accept_cancel()
    return accept.confirmed
