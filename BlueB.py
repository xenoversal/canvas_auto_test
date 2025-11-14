import sqlite3
import tkinter as tk
from tkinter import messagebox
from datetime import datetime, timedelta

DB_NAME = "my_exam_app.db"

def init_db():
    """Initialize the SQLite DB, create tables if not exist."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # Create a 'users' table with two columns: username and password
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    );
    """)
    conn.commit()
    conn.close()

def register_user_db(username, password):
    """Insert a new user into the database."""
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        # This occurs if the username is already taken (UNIQUE constraint)
        return False

def login_user_db(username, password):
    """Check if a user with given username & password exists in DB."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE username = ? AND password = ?", (username, password))
    row = c.fetchone()
    conn.close()
    return (row is not None)

# -------------------------------------------------------------------
# Exam Logic
# -------------------------------------------------------------------
class Exam:
    def __init__(self, questions, duration_minutes=5, username="guest"):
        self.questions = questions
        self.answers = [None] * len(questions)
        self.duration = timedelta(minutes=duration_minutes)
        self.start_time = None
        self.end_time = None
        self.submitted = False
        self.username = username

    def start(self):
        self.start_time = datetime.now()
        self.end_time = self.start_time + self.duration

    def time_left(self):
        if not self.start_time:
            return self.duration
        remaining = self.end_time - datetime.now()
        return max(remaining, timedelta(0))

    def is_over(self):
        return datetime.now() >= self.end_time

    def submit(self):
        self.submitted = True

    def save_results(self):
        """Save exam results to a file, named with username."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"exam_{self.username}_{timestamp}.txt"

        with open(filename, "w") as f:
            f.write(f"Exam Submission for {self.username}\n")
            f.write(f"Submitted at {datetime.now()}\n\n")
            for i, q in enumerate(self.questions, start=1):
                ans = self.answers[i-1] if self.answers[i-1] else "[No Answer]"
                f.write(f"Q{i} ({q['points']} pts): {q['text']}\n")
                f.write(f"Answer: {ans}\n\n")

        return filename

# -------------------------------------------------------------------
# Exam UI
# -------------------------------------------------------------------
class ExamUI(tk.Toplevel):
    def __init__(self, parent, exam: Exam):
        super().__init__(parent)
        self.title(f"{exam.username}'s Exam")
        self.geometry("800x600")
        self.resizable(False, False)

        self.exam = exam
        self.question_widgets = []
        self.radio_vars = []

        # Timer & instructions
        top_frame = tk.Frame(self, pady=5)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        self.timer_label = tk.Label(top_frame, text="", font=("Helvetica", 14), fg="red")
        self.timer_label.pack(side=tk.LEFT, padx=15)

        instruction_label = tk.Label(top_frame, text="Answer each question. Timer will auto-submit.", font=("Helvetica", 12))
        instruction_label.pack(side=tk.LEFT, padx=20)

        # Main Q&A frame
        self.qa_frame = tk.Frame(self, padx=10, pady=10)
        self.qa_frame.pack(fill=tk.BOTH, expand=True)

        for i, q in enumerate(self.exam.questions):
            q_label = tk.Label(self.qa_frame, text=f"Q{i+1} ({q['points']} pts): {q['text']}",
                               font=("Helvetica", 12, "bold"), wraplength=750, justify=tk.LEFT)
            q_label.pack(anchor="w", pady=(8, 2))

            if q["type"] == "sa":
                # Short answer
                text_box = tk.Text(self.qa_frame, height=3, width=90, wrap="word")
                text_box.pack(anchor="w", pady=(0, 10))
                self.question_widgets.append(text_box)
                self.radio_vars.append(None)
            else:
                # Multiple choice
                var = tk.StringVar(value="")
                self.radio_vars.append(var)
                self.question_widgets.append(None)
                rb_frame = tk.Frame(self.qa_frame)
                rb_frame.pack(anchor="w", pady=(0, 10))

                for choice in q["choices"]:
                    rb = tk.Radiobutton(rb_frame, text=choice, variable=var, value=choice,
                                        font=("Helvetica", 10), wraplength=700, justify=tk.LEFT)
                    rb.pack(anchor="w")

        # Submit button
        submit_btn = tk.Button(self, text="Submit Now", command=self.submit_exam,
                               bg="#007BFF", fg="white", font=("Helvetica", 12, "bold"))
        submit_btn.pack(pady=10)

        self.exam.start()
        self.update_timer()

        self.protocol("WM_DELETE_WINDOW", self.on_window_close)

    def update_timer(self):
        if self.exam.submitted:
            return
        remaining = self.exam.time_left()
        if remaining.total_seconds() <= 0:
            self.timer_label.config(text="Time is up!")
            self.auto_submit()
        else:
            mm, ss = divmod(remaining.total_seconds(), 60)
            self.timer_label.config(text=f"Time Left: {int(mm):02d}:{int(ss):02d}")
            self.after(1000, self.update_timer)

    def gather_answers(self):
        for i, q in enumerate(self.exam.questions):
            if q["type"] == "sa":
                text = self.question_widgets[i].get("1.0", tk.END).strip()
                self.exam.answers[i] = text
            else:
                selected = self.radio_vars[i].get()
                self.exam.answers[i] = selected

    def submit_exam(self):
        if self.exam.submitted:
            return
        self.gather_answers()
        self.exam.submit()
        filename = self.exam.save_results()
        messagebox.showinfo("Exam Submitted", f"Your exam has been submitted!\nSaved to: {filename}")
        self.quit_app()

    def auto_submit(self):
        if self.exam.submitted:
            return
        self.gather_answers()
        self.exam.submit()
        filename = self.exam.save_results()
        messagebox.showwarning("Time's Up", f"Time is up! Auto-submitting.\nSaved to: {filename}")
        self.quit_app()

    def on_window_close(self):
        if not self.exam.submitted:
            messagebox.showinfo("Cannot close", "You must submit your exam before closing.")
        else:
            self.quit_app()

    def quit_app(self):
        self.destroy()

# -------------------------------------------------------------------
# Registration Window
# -------------------------------------------------------------------
class RegisterWindow(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Create New User")
        self.geometry("300x220")
        self.resizable(False, False)
        self.parent = parent

        tk.Label(self, text="Username:").pack(pady=(10, 0))
        self.entry_user = tk.Entry(self)
        self.entry_user.pack()

        tk.Label(self, text="Password:").pack()
        self.entry_pass = tk.Entry(self, show="*")
        self.entry_pass.pack()

        tk.Label(self, text="Confirm Password:").pack()
        self.entry_pass2 = tk.Entry(self, show="*")
        self.entry_pass2.pack()

        tk.Button(self, text="Create", command=self.handle_create).pack(pady=10)

    def handle_create(self):
        username = self.entry_user.get().strip()
        password = self.entry_pass.get().strip()
        password2 = self.entry_pass2.get().strip()

        if not username or not password:
            messagebox.showerror("Error", "Username and password cannot be empty.")
            return

        if password != password2:
            messagebox.showerror("Error", "Passwords do not match.")
            return

        # Insert into SQLite
        success = register_user_db(username, password)
        if not success:
            messagebox.showerror("Error", f"Username '{username}' already exists.")
            return

        messagebox.showinfo("Success", f"User '{username}' created!")
        self.destroy()

# -------------------------------------------------------------------
# Login Window
# -------------------------------------------------------------------
class LoginWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Exam Login")
        self.geometry("300x220")
        self.resizable(False, False)

        tk.Label(self, text="Welcome! Please log in or create a new user.").pack(pady=10)

        tk.Label(self, text="Username:").pack()
        self.entry_user = tk.Entry(self)
        self.entry_user.pack()

        tk.Label(self, text="Password:").pack()
        self.entry_pass = tk.Entry(self, show="*")
        self.entry_pass.pack()

        tk.Button(self, text="Login", command=self.handle_login).pack(pady=5)
        tk.Button(self, text="Create New User", command=self.handle_create_user).pack()

    def handle_login(self):
        username = self.entry_user.get().strip()
        password = self.entry_pass.get().strip()

        # Check in SQLite
        if login_user_db(username, password):
            messagebox.showinfo("Login Success", f"Welcome {username}!")
            self.start_exam(username)
        else:
            messagebox.showerror("Login Failed", "Invalid username or password.")

    def handle_create_user(self):
        reg_win = RegisterWindow(self)
        reg_win.grab_set()  # modal window

    def start_exam(self, username):
        # Hide login window
        self.withdraw()

        # Example questions
        sample_questions = [
            {
                "text": "What does 'OOP' stand for?",
                "type": "sa",
                "choices": [],
                "points": 5
            },
            {
                "text": "Which of the following is a Python list method?",
                "type": "mc",
                "choices": ["add()", "remove()", "delete()", "pop()"],
                "points": 5
            },
            {
                "text": "Which data structure does BFS typically use?",
                "type": "mc",
                "choices": ["Stack", "Queue", "Priority Queue", "Linked List"],
                "points": 5
            },
        ]

        exam = Exam(sample_questions, duration_minutes=1, username=username)
        exam_ui = ExamUI(self, exam)

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Initialize DB & create tables
    init_db()

    app = LoginWindow()
    app.mainloop()
