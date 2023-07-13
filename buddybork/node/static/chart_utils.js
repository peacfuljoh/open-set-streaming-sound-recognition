/*
    Front-end code for handling Plotly chart info, plotting, etc.
*/


// plotly chart config
chartLayout = {
    title: {
        text: "",
        font: {family: "Times New Roman", size: 24}
    },
    xaxis: {
        title: {
            text: "Time",
            font: {family: "Courier New, monospace", size: 18, color: "#7f7f7f"}
        }
    },
    yaxis: {
        title: {
            text: "",
            font: {family: "Courier New, monospace", size: 18, color: "#7f7f7f"}
        }
    },
    margin: {t: "50", b: "50"}
}



// plot updates
function updatePlot(id_, data, layout, opts=null) {
    if (opts == null) { Plotly.newPlot(id_, data, layout); }
    else              { Plotly.newPlot(id_, data, layout, opts); }
}


// other
function getTagsAndColorsFromTagsInfo(tagsInfo, includeAlpha=false) {
    // output: array of tag strings, corresponding rgba color strings
    return [
        tagsInfo.tags.map(tag => tag.value),
        tagsInfo.colors.map(c => {
            let color = `rgba(${c[0]}, ${c[1]}, ${c[2]}, `;
            color += includeAlpha ? `0.3` : `1`;
            color += `)`;
            return color;
        })
    ]
}


