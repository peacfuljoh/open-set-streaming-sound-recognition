
const express = require('express');
const errorUtils = require('../utils/error_utils.js');

const rootRouter = express.Router();


const routerIndex = require('./routes_index.js');
const routerDashboard = require('./routes_dashboard.js');
const routerDetector = require('./routes_detector.js');
const routerAnnotator = require('./routes_annotator.js');
const routerMeta = require('./routes_meta.js');
const routerTrainer = require('./routes_trainer.js');
const routerRaw = require('./routes_raw.js');
const routerDataCapture = require('./routes_data_capture.js');
const routerLogout = require('./routes_logout.js');

const loginUtils = require('../utils/login_utils.js');


rootRouter.use('/', routerIndex);
rootRouter.use('/dashboard', routerDashboard);
rootRouter.use('/detector', (req, res) => loginUtils.sessionHandler(req, res, routerDetector));
rootRouter.use('/annotator', (req, res) => loginUtils.sessionHandler(req, res, routerAnnotator));
rootRouter.use('/meta', (req, res) => loginUtils.sessionHandler(req, res, routerMeta));
rootRouter.use('/trainer', (req, res) => loginUtils.sessionHandler(req, res, routerTrainer));
rootRouter.use('/raw', (req, res) => loginUtils.sessionHandler(req, res, routerRaw));
rootRouter.use('/dataCapture', (req, res) => loginUtils.sessionHandler(req, res, routerDataCapture));
rootRouter.use('/logout', routerLogout);

rootRouter.all('*', (req, res) => errorUtils.checkErr(req, res, false, 'router_root'));


module.exports = rootRouter;

