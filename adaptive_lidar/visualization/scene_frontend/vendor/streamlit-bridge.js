// Minimal Streamlit custom-component bridge.
//
// Streamlit's official component library is a React/TypeScript build. This
// page has no build step on purpose — it is four static files served out of
// the repository so the app runs offline from a fresh clone — so the three
// postMessage calls the protocol actually needs are written out here.
//
// Protocol, as implemented by Streamlit's iframe host:
//   streamlit:componentReady    once, when the page can accept a render
//   streamlit:setFrameHeight    whenever the iframe should resize
//   streamlit:setComponentValue the value Python receives back
// and inbound "streamlit:render" carries {args, disabled, theme} in
// event.data.

(function () {
  function post(type, data) {
    window.parent.postMessage(
      Object.assign({ isStreamlitMessage: true, type: type }, data), "*");
  }

  window.Streamlit = {
    RENDER_EVENT: "streamlit:render",
    setComponentReady: function () {
      post("streamlit:componentReady", { apiVersion: 1 });
    },
    setFrameHeight: function (height) {
      post("streamlit:setFrameHeight", { height: height });
    },
    setComponentValue: function (value) {
      post("streamlit:setComponentValue", { value: value, dataType: "json" });
    },
    events: {
      addEventListener: function (type, callback) {
        window.addEventListener("message", function (event) {
          if (event.data && event.data.type === type) {
            event.detail = event.data;
            callback(event);
          }
        });
      },
    },
  };
})();
