/*
    Back-end session handling utils.
*/

constants = require('../constants_app.js');

const expressSession = require('express-session');
const pgSession = require('connect-pg-simple')(expressSession);

const { Pool, Client } = require('pg');
const pgPool = new Pool(constants.DB_CONFIG);


function createSession(secure) {
    const cookie = {
        maxAge: 1 * 60 * 60 * 1000,
        secure: secure
    }

    const sess = expressSession({
      store: new pgSession({
        pool: pgPool,
        tableName : 'session'
        // Insert connect-pg-simple options here
      }),
      saveUninitialized: false,
      secret: process.env.SESSION_SECRET,
      resave: false,
      cookie: cookie
      // Insert express-session options here
    })

    return sess;
}


module.exports = {
    createSession
}

