
const { Pool, Client } = require('pg');

const constants = require('../constants_app.js');


const pool = new Pool(constants.DB_CONFIG);

function limQuery(q, trim=null) {
    const trimLen = (trim == null) ? constants.MAX_QUERY_LOG_LEN : Math.min(trim, constants.MAX_QUERY_LOG_LEN);
    if (q.length > trimLen) { return q.substring(0, trimLen) + '...'; }
    else                    { return q; }
}

function logQuery(q, trim=null) {
    console.log(limQuery(q, trim));
}

function selectQuery(data, table, condition='', trim=null) {
    return new Promise((resolve, reject) => {
        let q = `SELECT ${data} FROM ${table} ${condition}`;
        logQuery(q, trim);
        pool.query(q, (err, result) => {
            if(err) { reject(err); }
            if(result === undefined) { resolve([]); }
            resolve(result.rows);
        })
    });
}

function insertQuery(table, cols, values) {
    let q = `INSERT INTO ${table} ${cols} VALUES ${values}`;
    return pool.query(q)
        .then(res => {
            logQuery(q);
            return q;
        })
        .catch(err => console.log('insertQuery failed on: ', limQuery(q), '\n', err));
}

function updateQuery(table, data, condition) {
    let key = Object.keys(data)[0];
    let val = Object.values(data)[0];
    let q = `UPDATE ${table} SET ${key} = '${val}' WHERE ${condition}`;
    return pool.query(q)
        .then(res => {
            logQuery(q);
            return q;
        })
        .catch(err => console.log('updateQuery failed on: ', limQuery(q), '\n', err));
}

function deleteQuery(table, condition) {
    let q = `DELETE FROM ${table} WHERE ${condition}`;
    return pool.query(q)
        .then(res => {
            logQuery(q);
            return q;
        })
        .catch(err => console.log('deleteQuery failed on: ', limQuery(q), '\n', err));
}


module.exports = {selectQuery, insertQuery, updateQuery, deleteQuery};