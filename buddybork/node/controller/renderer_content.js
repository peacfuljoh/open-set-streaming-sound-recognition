

/* Help text to be displayed on each page */
const helpTextIndex = `
    <h2>How to use the Dashboard</h2>

    <h3>Monitoring sound events in real-time</h3>
    <p>
        Monitor sound events in real-time by selecting one of the pre-trained models from the dropdown menu on
        the left. This will display a chart of detections (by tag) over the recent past. This chart will update
        with new detections every 60 seconds.
    </p>
    <p>
        Once the detection chart is available, you can click on any detection point to see the raw waveform
        at that point in time. This 15-second segment of audio is called a macrosegment. You can also listen
        to the audio itself.
    </p>

    <h3>Controlling the data stream</h3>
    <p>
        The slider on the right allows you to toggle the data stream on and off.
    </p>
`;

const helpTextAnnotator = `
    <h2>How to use the Annotator</h2>

    <h3>Inspecting the database</h3>
    <p>
        This page includes four charts for inspecting and annotating the database at different scales, each
        subsequent one providing a finer-grained view:
    </p>
    <ol>
        <li>
            <strong>Database summary</strong>: This chart shows how many 3-second segments of raw data were
            captured for each day in the dataset as a stem plot. Clicking on a point will bring up
            that day's segment scatterplot (see chart 3's description below).
        </li>
        <li>
            <strong>Annotation timeline</strong>: This chart provides a color-coded timeline
            summary of all annotations in the
            database. Clicking on a point will bring up the macrosegment centered at that annotation (see
            chart 4's description below).
        </li>
        <li>
            <strong>Segment scatterplot</strong>: This chart provides a scatter plot of all 3-second segments for a
            single day where the vertical position corresponds to the largest amplitude (in decibels) of the signal
            within a segment. In addition, annotations are overlaid as vertical lines. This provides a
            fast way to find sounds of interest by manual inspection. Clicking on a point brings up the macrosegment
            centered on the selected segment.
        </li>
        <li>
            <strong>Macrosegment plot</strong>: This plot provides the finest-level data visualization corresponding
            to the signal waveform itself. Clicking on a point in charts 2 or 3 will bring up the corresponding
            15-second window of data around that point. Annotations are displayed on this chart and can be inserted
            as described below.
        </li>
    </ol>

    <h3>Editing annotations</h3>
    <p>
        Controls in the macrosegment plot allow for navigating left and right in time as well as inserting/removing
        annotations. A dropdown menu allows for selecting the tag to use for inserted annotations. All annotations
        will be overlaid on top of the waveform exactly as they are found in the database. Thus, there is no save/undo
        feature. One simply modifies the existing annotations as needed.
    </p>
`;

const helpTextTrainer = `
    <h2>How to use the Model Builder</h2>

    <p>
        This page enables training and testing of machine learning (ML) models by dividing up the dataset by day into
        train and test sets. The segmented bar graph shown provides a high-level summary of the annotations available
        across all days in the dataset. The legend on the right-hand side can be used to view a subset of the tags
        at a time or even just view one tag (click or double-click on a legend item).
    </p>

    <h3>Training a model</h3>
    <h4>Select days for the training set</h4>
    <p>
        To train a model, you first select what days to use in the training set. This can be done by selecting the
        <i>Train (R)</i> option from the "Grouping Mode" dropdown menu and clicking on the bar graph for the
        desired days. The more days are included in the training set the better as this will provide more
        annotated examples of the types of sounds to be recognized.
    </p>
    <h4>Select days for training the background model</h4>
    <p>
        Next, you can select one or more days to use for training a background model with the
        <i>Background Train (Br)</i> option. This is highly recommended as it will result in much better
        performance (one or two days is sufficient).
    </p>
    <h4>Select tags</h4>
    <p>
        Then, you check off which tags you want to include in this new model via the <i>Training Tags Select</i>
        checklist.
    </p>
    <h4>Fit the model</h4>
    <p>
        Once the desired days and tags are selected for training, hit the <i>Fit model</i> button and
        give the model a unique name. This will initiate a model training procedure
        (including feature extraction) that typically takes no more than 30 seconds to complete.
    </p>

    <h3>Testing a model</h3>
    <h4>Select days for the test set</h4>
    <p>
        To test a pre-trained model, select the days with annotations that you want to include in a test set with the
        <i>Test (E)</i> option. You similarly select days to include randomly-chosen segments from with the
        <i>Background Test (Be)</i> option. This chooses segments at random from these days and is a good way
        to verify that the model correctly rejects out-of-class examples.
    </p>
    <h4>Evaluate with a pre-trained model</h4>
    <p>
        Once days with annotations and background examples are selected, you select a pre-trained model from the
        <i>Select model</i> dropdown menu and hit <i>Predict with model</i>. This initiates feature
        extraction and model evaluation for this test set and typically takes no more than 30 seconds to complete.
        Upon completion, a confusion matrix will appear showing how many examples of each sound in the test set
        were classified under each tag in the training set.
    </p>

    <h3>Deleting a model</h3>
    <p>
        To delete a (unused/outdated) model, select it from the <i>Select model</i> dropdown menu and double-click
        the <i>Delete model</i> button. Verify that you are sure you want to do this because models cannot be recovered
        once deleted. However, they can be retrained relatively easily.
    </p>

    <h3>Helpful tips</h3>

    <h4>Low detection rate</h4>
    <p>
        Some sounds are simply harder to recognize, but if the model
        performance doesn't look good at first for some tags, there could be multiple reasons that don't have
        to do with features of the sound itself.
    </p>
    <p>
        One common cause
        is a lack of training data relative to the largest-count tag in the training set. The model takes into account
        relative proportions of annotations, so a rare tag is more likely to be classified as "other".
    </p>
    <p>
        Also, part of the model captures time-of-day information, so if the train and test sets cover non-overlapping
        times for a given tag, the model will struggle on that tag.
    </p>
    <p>
        And finally, if no days are selected for background training data,
        the model is much more limited in how it can set detection thresholds. So be sure to include one or two
        days for this!
    </p>

    <h4>Feature caching</h4>
    <p>
        Features are not pre-extracted and so training and
        testing on previously unused days will take longer at
        first. However, once features have been computed, they are fetched from the server's cache on subsequent
        runs, leading to faster train/test times. This is true more for annotations than for background examples,
        which are selected at random, but even those will be cached more fully as you perform more train/test runs.
    </p>
    <p>
        This delay is not related to the speed at which the model evaluates a new audio
        segment. The model is designed to evaluate on new examples many times faster than real-time
        to enable streaming sound recognition.
    </p>
`;

const helpTextLogin = `
    <h2>Welcome</h2>

    <h3>What is this?</h3>
    <p>
        This website is the result of a learning project to create a full-stack, open-set sound recognition (OSSR) system.
        "Open-set" indicates that it is able to distinguish between human-annotated classes of sounds while also understanding
        when it hasn't heard a sound before. This is an important property of any recognition system that is deployed
        "in the wild".
    </p>
    <p>
        The system consists of various components: audio data collection on an edge device, search and annotation of
        sounds, training and testing of models, and streaming detection of events.
    </p>

    <h3>How does it work?</h3>
    <p>
        The web application you are currently interacting with is served from a Node.js app running on an Ubuntu server
        that has been made accessible through the internet.
        A data lake consisting of a PostgreSQL database, a feature store, and a model store is hosted on this server.
        A separate Flask app controls an edge device (Raspberry Pi) that is set up for capturing audio
        in real time. All offline and real-time functionality of the system is controlled through the pages of this
        web app.
    </p>

    <h3>What does the name mean?</h3>
    <p>
        This system was originally set up to capture data in a domestic setting. As such, the most
        easily recognizable event in the database turned out to be dog barks. Translating this into Doggolingo (the
        internet slang for describing dog behavior), we get "buddybork". That being said, the system does implement
        general-purpose recognition.
    </p>
    <p>

    </p>
`;



module.exports = {
    helpTextAnnotator,
    helpTextIndex,
    helpTextLogin,
    helpTextTrainer
}