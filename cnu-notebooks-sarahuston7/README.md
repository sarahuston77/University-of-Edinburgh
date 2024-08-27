# Computing and Numerics - Course notebooks

The `.ipynb` files in each folder are your **course notebooks**. They are your course notes/materials, and also contain interactive code examples and exercises for you to review in your own time.

All notebooks for the entire semester are available from Week 1 -- but do pace yourself! There are 10 folders, from Week 1 to Week 10, each containing one notebook. By the end of each week, before your Friday workshop slot, you are expected to have reviewed the corresponding weekly notebook.


## How do I get started?

### To start a codespace (cloud environment):

- Click on the green "<> Code" button (top right).
- Click on the "Codespaces" tab.
- Click on the green "Create codespace on main" button.
- Wait a few minutes for everything to get set up.
- Once VSCode appears in your browser, click on the `w01` folder, then `week01.ipynb` in the left bar to start your notebook.

If you have issues running code in the notebook, we strongly recommend that you use Google Chrome or Chromium. Some notebook features may not work in other browsers.

When you come back to the notebook later, you'll be able to rejoin your previous codespace (instead of creating it again). GitHub will give it a random goofy name, and it will show up instead of the "Create codespace on main" button.

Codespaces are kept for 30 days; if you don't come back to a codespace for 30 days, it will be deleted, and you will lose your work. In the Week 1 workshop, you will learn how to upload your changes back to GitHub, so that you don't lose them.

### If you prefer working offline:

You need to clone this repository. This is how (assuming you've installed all the required software):

- Click on the green "<> Code" button (top right).
- Click on the "Local" tab.
- Copy the URL given to you.
- Launch VSCode.
- In VSCode, press `Ctrl + Shift + P` (`Cmd + Shift + P` on MacOS), search for "clone", and select "Git: Clone".
- Paste the URL in the text box and press Enter.
- Select a folder where you want to store the files.
- If required, authenticate with GitHub.
- Finally, the files should appear in VSCode. Click on the `w01` folder, then `week01.ipynb` in the left bar to start your notebook.


## What are the other files?

You don't need to touch any of them, but if you're wondering:

- `README.md` is this very file.
- `.gitignore` is a simple list of everything we don't want git to track.
- In each folder, `show_solutions.py` will be used inside your notebook, for you to reveal exercise solutions when they are released. You don't need to run it separately -- just follow the instructions inside the notebooks.
- `.devcontainer` contains configuration files, which are used to set up your codespace with the tools you will need.
