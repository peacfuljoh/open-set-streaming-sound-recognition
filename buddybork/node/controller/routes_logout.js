
const express = require('express');

const router = express.Router();
const renderer = require('./renderers.js');

router.get('/', (req, res) => {
    delete req.session.userid;
    renderer.render(res, 'login');
});

router.all('/*', (req, res) => errorUtils.checkErr(req, res, false, 'router_logout'))

module.exports = router;