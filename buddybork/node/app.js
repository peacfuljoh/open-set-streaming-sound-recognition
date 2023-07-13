/* Entry point for Buddybork app (Node.js/Express) */

const PROD = false; // this should go in an environment variable --> app.get('env') === 'production';
                    // currently breaks session handling (doesn't remember user id in cookie...?)

require('dotenv').config(); // load config environment variables

// includes
const http = require('http');
const path = require("path");
const express = require("express");
const bodyParser = require('body-parser');

const router = require('./controller/routes_root.js');
const constants = require('./constants_app.js');
const sessUtils = require('./utils/sess_utils.js');


// app description
const app = express();

// session handler
if (PROD) { app.set('trust proxy', 1); } // trust first proxy
app.use(sessUtils.createSession(PROD));

// views
app.set('view engine', 'pug');
app.set("views", path.join(__dirname, "views"));

// security features
if (0) {
    const compression = require("compression");
    const helmet = require("helmet");
    app.use(helmet()); // reduce common security threats
    app.use(compression()); // compress all routes
}

// middleware stack
app.use(bodyParser.urlencoded({extended: false})); // preprocesses request body into object at req.body
app.use('/static', express.static('./static')); // allows use of CSS and other static files
app.use('/assets', express.static(constants.ASSETS_DIR)); // assets dir has to be at level of app or below, otherwise client-side fetch won't work
app.use('/', router);

// app start
app.listen(constants.SERVER_PORT, constants.IP_ADDRESS, () => {
  console.log(`Application served at ${constants.IP_ADDRESS}:${constants.SERVER_PORT}`);
});
