const express = require('express');

const loginUtils = require('../utils/login_utils.js');
const renderers = require('./renderers.js');

const router = express.Router();


// upon submitting login form
router.post('/', (req, res) => {
    const name = req.body.username;
    const pw = req.body.password;
    loginUtils.verifyCredentials(name, pw)
        .then(valid => {
            if (valid) {
                req.session.userid = name; // save verified username in session cookie
                renderers.render(res, 'index');
            }
            else { renderers.render(res, 'login'); }
        });
});

router.get('/', (req, res) => {
    loginUtils.sessionHandler(req, res, () => { renderers.render(res, 'index'); });
});





module.exports = router;
