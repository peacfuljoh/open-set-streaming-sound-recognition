# Open-set recognition system (OSR) for real-time sound event detection


## Overview

This project (codename Buddybork) is my effort to learn more about full-stack development (running a secure web server, 
serving an interactive web app to a client browser), database management, and algorithm integration 
(batch train/test, real-time detection).

This repository contains code for the five components of this OSR system:
 
- Continuous audio capture and logging (Python)
- Web server running a Node app (Node/Javascript)
- Client-side Pug templates and other static resources comprising the user-facing web app (HTML, Javascript)
- SQL database that stores metadata on raw data segments, annotations, tags, etc.
- Offline ML dev tools

Some configuration and setup is required outside of this code:

- Setting up the SQL database and necessary tables (see below).
- Configuring repo and data roots, user info, database info, etc.

Further setup is required to upgrade the web server to production (see below).


## Data capture

When the data capture Python process is running, it continuously pulls audio from a microphone through an audio 
interface at 44.1 kHz and chunks it into 3-second segments. If the segment contains non-silence, it is timestamped and 
stored in two parts:

1. Raw data: on disk as a .wav file.
2. SQL table entry: as an entry in the `raw` table w/ (datetime, max_amp, filepath) info.

The `max_amp` value is the largest absolute sample magnitude in the segment, which is used for general-purpose 
filtering as well as making it easier for a client-side user to find sounds to annotate.

The code for this can be found in the `src/` directory. The script to run on-site is `src/main.py`, which uses database-access
utils available in `src/utils/db_utils.py` and `src/models/db_models.py`. This only requires that
the `raw` table exist and can be run independently of the other system components. Configuration info is in 
`src/constants_stream.py`.

Use `make stream` to run the stream process (see "Automation via make"). The data
stream itself is toggled from the web app dashboard page and the capture state is maintained via the JSON file `data_capture_state.json`. This file is local to the server (not in the main repo) and the path to it is stored in `paths.json`.


## Web server

The web server code is entirely contained in the `node/` directory. Configuration info is in `node/constants_app.js`. 
The code is arranged in a Model-View-Controller (MVC) architecture:

- Entry point: the entry point for the Node.js (Express) app is `node/app.js`.
- Model: `node/db/` has general-purpose code for issuing queries to the SQL database.
- View: `node/views/` contains Pug templates for rendering client-side HTML.
- Controller: `node/controller/` contains a heirarchy of Express routes.

Other components are:

- ML subprocesses: any machine learning-related computations (train, test, detect) are handed over to Python processes with 
entry-point code found in `node/py/`. These scripts then call feature extraction and model code in `node/ml/`.
- Static resources: static resources (e.g. Javascript, CSS) that are served along with HTML can be found in `node/static/`.
- Utility functions: Javascript modules containing helper functions for different parts of the app are contained in `node/utils/`. An important util file is `fetch_utils.js`, which describes all of the routes for interactions between the frontend and backend.

This is all that is required to run the app in development mode. It will not be accessible outside of your LAN. To move
the app to production, see the section below.

User sessions are managed through `express-session` using `connect-pg-simple` for the data store option, so these need 
to be setup as described in their readmes.


## Database

The database has a few components split between files in a data directory and entries in SQL tables.

### Data files

The files are:
- Raw captured audio (.wav files): this is the only type of file that cannot be re-generated.
- Cached features: these contain pre-computed feature vectors and metadata for annotated and raw segments.
- Saved models: a JSON file contains an index into the current library of models (one file each).

Raw data files are named by their UTC time in milliseconds.

### SQL tables

The SQL tables are:
- "raw": index into raw captured audio, (datetime, max_amp, filepath)
- "annots": user-provided annotations, (datetime_start, datetime_end, tag)
- "tags": simple list of strings of all available tags (tag)
- "detections": cache of detection info for fast update/display of real-time detection chart (datetime, tag)
- "login": user credentials to enter through landing page

The database setup is pretty easy through the command line in Linux. A convenient way to inspect SQL tables directly 
with queries is through the Linux CLI, e.g. `psql buddybork`.


## Developer tools

All ML-related dev work can be done through Python code in `src/`.

### Scripts

The `src/scripts/` directory has many high-level 
scripts for training models and viewing extracting raw waveforms, etc. The main script for train/test model development
is `src/scripts/train_test_main.py`. The `src/scripts/fit_time_model.py` script visualizes the time models for all tags
in the database all at once. There are other miscellaneous scripts for debugging, patching the DB during dev, etc. They
can be ignored.

### ML tools

To extract features, methods in `src/utils/feats_io_utils.py` are called, which construct an instance of the `FeatureSet`
class, defined in `src/ml/features.py`. The actual featurization code is in `src/ml/featurization.py`.

These features can then be used to train the recognition model defined in `src/ml/ml_models.py`, which incorporates both
a linear projector in `src/ml/transformers.py` and a classifier in `src/ml/classifiers.py`.

Various helper functions are divided by function throughout the files in `src/utils/`.

There are also a few plotting utilities that can be manually turned on to view annotations' raw data and extracted 
features (see `src/ml/featurization.py`) and training set precision-recall curves (see `src/ml/classifiers.py`) when 
background training data is provided.


## Web app usage

All instructions are included in help dialogs on each page of the web app, but here's an overview anyway.

### Login and dashhboard

When a user accesses the landing page (i.e. http://<WAN_IP_ADDRESS>:<PORT> or https://www.hasthebuddyborked.com), they are taken to a login screen. Upon entering credentials that match one in the `login` SQL table, a session is established and they are served the main dashboard.

The dashboard can be used for real-time detection and monitoring of sound events using a pre-trained of choice.

### Annotator

Navigating to the annotator page displays a day-by-day graph of raw segment counts captured by the data collection system as well as a color-coded chart of all annotations created thus far (including per-tag counts). Clicking on points in either chart will bring up other, more fine-grained charts: day-specific segment info and the macrosegment plot. The former shows `max_amp` values for captured segments as well as vertical lines for each annotation for the duration of one day. The latter show the raw audio for a 15-second window centered on the selected segment or annotation. The macrosegment plot is where manual annotations can be created or removed, upon which the change will be propagated to all other affected plots.

### Model builder

Navigating to the model builder page displays a segmented bar graph with annotation counts for each day available in the database. This is useful for seeing at a high level where annotations are available on a day-by-day basis.

Controls on this page can be used to divide days into train and test sets, including which days are to be used as background data in either set. A subset of the full list of tags can be selected during training, with each subset corresponding to a new model fit. Background data in the train set is used to set detection thresholds for a precision-recall trade-off. Whereas background data in the test set is used to evaluate the model on unannotated segments. Both of these are important for developing a high-quality open-set recognition model.

A trained model can be run on the test set, the results being displayed as a confusion matrix.




## Other info

### Setup conda environment

The `environment.yml` file can be used to create a complete conda environment via `conda env create -f environment.yml`.

### Install Javascript dependencies

Then install Node (v19.7.0) and the Node Package Manager (npm) and run `npm install` in the repo root. This will install all the Javascript dependencies listed in `package.json`.

### Automation via make

All CLI ops are automated via make:

- `make build` sets up environment variables
- `make app` runs the Node app
- `make stream` runs the data collection process
- `make cleanup` runs the cleanup process
- `make test` runs all tests (Javascript, Python)

See `Makefile` in the repo root for the shell recipes.

### Continuous Integration

GitHub Actions is used for CI/CD. See `.github/workflows/buddybork.yml` for the testing recipe that gets executed on push.

### Manual backups

Manual backups can be easily created and kept updated with the script `src/scripts/update_data_backup.py`. Only raw data
files and the SQL tables are backed up.

### Autostart processes

To avoid having to start continuous processes (e.g. data capture, web server), you can use autostart functionality 
like systemd in Linux. For a simple and otherwise reliable setup, this isn't really necessary. If you don't have
autostart functionality set up, keep the startup commands/instructions listed somewhere for easy reference. Most are handled by `make` commands listed in the repo's Makefile (except for Nginx, which after setup just needs `sudo systemctl start nginx`).

### Node.js app dev

A convenient way to develop a Node.js app is to use a responsive CLI that restarts the app whenever it detects
changes in the source code, e.g. `nodemon npx app.js`. This allows `console.log()` statements from Javascript code
and `print(); sys.stdout.flush()` statements in Python subprocesses that are handled by the Javascript caller 
(see `runPyOp()` in `node/utils/py_utils.js` and how `print_flush()` is used in e.g. `node/py/pydetection.py`) 
to print to the console for debugging purposes.

### Asset directory

Whenever a user selects a segment in the database through the client-side web app, a macroseg is patched together on the server, a temporary .wav file is created with the macroseg audio, and its filepath returned. This asset folder is statically served alongside the other static files (eg. .js, .css). The .wav file is then accessed client-side with a fetch command.

The simple Python script `main_asset_cleanup.py` (run via `make cleanup`) can be run in the background to clean out this directory once a day.

### Access Node endpoints from command line

First, submit credentials through the dashboard route and store the resulting cookie locally:

`curl -X POST https://www.hasthebuddyborked.com/dashboard -H "Content-Type: application/x-www-form-urlencoded" -d "username=<USERNAME>&password=<PASSWORD>" --cookie-jar temp`

Then, use the cookie to access any of the endpoints directly:

`curl -X GET https://www.hasthebuddyborked.com/meta --cookie temp | jq`

The pipe into `jq` takes the output of the `curl` command and displays it (JSON data) nicely (like `pprint` in Python).

### External repos

At some point during development, several modules were sectioned off (e.g. `ossr_utils`), packaged using Poetry, and published to PyPI so that they can be installed from anywhere (including a GitHub Actions workflow) via `pip install <package_name>`. The process for setting this up is described here: https://py-pkgs.org/03-how-to-package-a-python.

### Running data capture on a Raspberry Pi

The original data capture code was divided into one that runs a local Flask server (`data_capture_flask_app.py`) and code running on an edge device (e.g. Pi) from a split-off repo `ossr-capture`. This allows data capture to be untethered from the main server device.

Setup for the server: run the Flask app (`make stream-flask`) in the main repo.

Setup for the Pi:

Use the RPi Imager to install Raspberry Pi OS (64 bit) on the SD card. Go through Pi setup, install miniconda, clone the `ossr_capture` repo, `make build` (with CLI args specifying the non-server root dir), and then `make stream`. You can also automate the Python process via `systemd` (https://linuxhandbook.com/create-systemd-services/amp/) using the `.system` file in the `ossr-capture` repo. If miniconda isn't working, use pip to install the few required packages manually into the base environment.

Make sure to install dependencies (ALSA, PortAudio) for `sounddevice`:

sudo apt-get install libasound-dev
sudo apt-get install libportaudio2

Note: once you have everything set up, collect some speech and inspect it to make sure there aren't glitches. If there are glitches at segment boundaries, it could have to do with the Portaudio version, how Python is managing thread activity, etc. It will need some debugging. Little glitches introduce noise that can go largely undetected by a recognition model but still impacts the overall functioning of the system (!). So be sure to triple-check the quality of the audio you're capturing after stitching the segments together.

### Local config

Config files with db and network secrets are stored on the server device (a.k.a. hub) but outside of the main repo. These are:

- db_config.json: host ('local_host'), port, user, password, database, max (workers in pool), dialect ('postgresql')
- net_config.json: HUB_IP, HUB_NODE_PORT, HUB_DATA_CAP_FLASK_PORT

For a new deployment, paths to these need to be updated in the `paths.json` config file.



## Deploying the app for production

Once the app is running in development mode (stand-alone Node.js app) and is accessible within your LAN via 
`http://<DEVICE_IP>:<PORT>`, a few more steps get the app on the internet in "production" mode.

1. Setup port forwarding through your LAN router to expose your server device's address to the internet (port 80 for HTTP and port 443 for HTTPS).
2. Buy a domain name with a DNS provider (e.g. Namecheap) and enable rerouting from this domain to your server device address via WAN_IP:PORT.
3. Set up and configure Nginx as a reverse proxy server (as well as PM2 for persistence, if desired). For full setup instructions on Ubuntu 20.04, see https://www.digitalocean.com/community/tutorials/how-to-set-up-a-node-js-application-for-production-on-ubuntu-16-04. After setup, start nginx via `sudo systemctl start nginx`.
4. Make sure to enable (free) SSL certification through Nginx so that all traffic between client and server is encrypted.

The MDN website has useful info about switching from dev to prod: https://developer.mozilla.org/en-US/docs/Learn/Server-side/Express_Nodejs/deployment.


