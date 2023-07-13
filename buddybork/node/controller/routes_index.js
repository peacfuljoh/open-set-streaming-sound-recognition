
const express = require('express');

const loginUtils = require('../utils/login_utils.js');
const renderers = require('./renderers.js');

const router = express.Router();


// landing page, login screen
router.get('/', (req, res) => {
    loginUtils.sessionHandler(req, res, (req, res) => renderers.render(res, 'index'))
});


module.exports = router;
