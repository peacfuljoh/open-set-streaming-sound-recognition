/*
    Back-end credential processing utils.
*/

const miscUtils = require('../utils/misc_utils.js');
const db = require('../db/database_utils.js');
const renderer = require('../controller/renderers.js');


function verifyCredentials(name, pw) {
    const pwHash = miscUtils.hashString(pw);
    const condition = `WHERE username = '${name}' AND password = '${pwHash}'`;
    const queryLen = 55;
    const trim = queryLen + name.length
    return db.selectQuery("*", "login", condition, trim)
        .then(result => (result.length == 1));
}

function validCred(req) {
    if (req.session.userid) {
        return true;
    }
    else {
        return false;
    }
}

function sessionHandler(req, res, next) {
    if (validCred(req)) { next(req, res); }
    else                { renderer.render(res, 'login'); }
}

module.exports = {
    verifyCredentials,
    sessionHandler
}
