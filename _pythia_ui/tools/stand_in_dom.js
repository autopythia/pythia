// tools/stand_in_dom.js: a stand-in DOM, for testing glem.js's view and
// glui.js's recipes in a JavaScript engine without a browser: elements with
// attributes, children, listeners and a style, text, and the keyboard's
// focus; what the view and the recipes use, and no more. It counts the
// elements that it makes, so that tests can tell which ones stay.
var standInDom = (function () {
  "use strict";

  function Document() {
    this.activeElement = null;
    this.made = 0;
    this.title = "";
    // The elements near the screen (near), and the observers that watch
    // for them, as a browser's IntersectionObserver does, from the window;
    // and the elements scrolled into view, in order.
    this.nearby = new Set();
    this.revealed = [];
    this.observers = [];
    var doc = this;
    this.defaultView = {
      IntersectionObserver: function (callback) {
        return new Observer(doc, callback);
      }
    };
  }

  // A stand-in IntersectionObserver: it tells its callback whether each
  // element that it starts to watch is near the screen, at once, where a
  // browser's tells it soon after; and then each time that changes.
  function Observer(doc, callback) {
    this.doc = doc;
    this.callback = callback;
    this.watched = new Set();
    doc.observers.push(this);
  }

  Observer.prototype.observe = function (element) {
    this.watched.add(element);
    this.callback([{ target: element, isIntersecting: this.doc.nearby.has(element) }], this);
  };

  Observer.prototype.unobserve = function (element) {
    this.watched.delete(element);
  };

  Observer.prototype.disconnect = function () {
    this.watched.clear();
  };

  // An element comes near the screen, or goes from it: each observer that
  // watches it hears so.
  Document.prototype.near = function (element, near) {
    if (near) {
      this.nearby.add(element);
    } else {
      this.nearby.delete(element);
    }
    this.observers.forEach(function (observer) {
      if (observer.watched.has(element)) {
        observer.callback([{ target: element, isIntersecting: near }], observer);
      }
    });
  };

  Document.prototype.createElement = function (tag) {
    this.made++;
    return new Element(this, tag);
  };

  Document.prototype.createTextNode = function (data) {
    return new Text(this, data);
  };

  // The events that don't bubble, as a browser fires them.
  var ALONE = ["toggle", "focus", "blur"];

  // An event, as the tests make them, on `target`: its listeners, then,
  // unless it's one that doesn't bubble or a listener stops it, its
  // ancestors'. `init` adds to the event, as {key: "Escape"}. A click on a
  // disabled control fires nothing, as in a browser.
  Document.prototype.fire = function (target, type, init) {
    if (type === "click" && disabled(target)) {
      return null;
    }
    var event = {
      type: type,
      target: target,
      defaultPrevented: false,
      stopped: ALONE.indexOf(type) >= 0,
      preventDefault: function () {
        event.defaultPrevented = true;
      },
      stopPropagation: function () {
        event.stopped = true;
      }
    };
    Object.keys(init || {}).forEach(function (name) {
      event[name] = init[name];
    });
    for (var node = target; node; node = event.stopped ? null : node.parentNode) {
      (node.listeners[type] || []).slice().forEach(function (listener) {
        listener(event);
      });
    }
    return event;
  };

  // The keyboard's focus moves to an element, or away (null): focusout on
  // the one that had it, then focusin, as a browser fires them. A disabled
  // control takes no focus.
  Document.prototype.focus = function (element) {
    var old = this.activeElement;
    if (old === element || (element && disabled(element))) {
      return;
    }
    this.activeElement = element;
    if (old) {
      this.fire(old, "focusout", { relatedTarget: element });
    }
    if (element) {
      this.fire(element, "focusin", { relatedTarget: old });
    }
  };

  // Whether an element is a control that's disabled.
  function disabled(element) {
    var controls = ["button", "input", "select", "textarea"];
    return controls.indexOf(element.localName) >= 0 && element.getAttribute("disabled") !== null;
  }

  // Text, in an element.
  function Text(doc, data) {
    this.ownerDocument = doc;
    this.nodeType = 3;
    this.data = String(data);
    this.parentNode = null;
    this.childNodes = [];
    this.listeners = {};
  }

  Text.prototype.contains = function (node) {
    return node === this;
  };

  function Element(doc, tag) {
    this.ownerDocument = doc;
    this.nodeType = 1;
    this.tagName = tag;
    this.localName = tag;
    this.parentNode = null;
    this.childNodes = [];
    // {name, value}, in the order that they're first set, as a browser's.
    this.attributes = [];
    this.listeners = {};
    this.style = new Style();
  }

  Element.prototype.insertBefore = function (child, ref) {
    if (child.contains(this)) {
      throw new Error("insertBefore would put an element inside itself");
    }
    if (child.parentNode !== null) {
      child.parentNode.removeChild(child);
    }
    var at = ref === null || ref === undefined ? this.childNodes.length : this.childNodes.indexOf(ref);
    if (at < 0) {
      throw new Error("insertBefore's reference isn't a child");
    }
    this.childNodes.splice(at, 0, child);
    child.parentNode = this;
    return child;
  };

  Element.prototype.appendChild = function (child) {
    return this.insertBefore(child, null);
  };

  // Removes a child. The focus goes with it, without an event, as a
  // browser's does.
  Element.prototype.removeChild = function (child) {
    var at = this.childNodes.indexOf(child);
    if (at < 0) {
      throw new Error("removeChild's node isn't a child");
    }
    this.childNodes.splice(at, 1);
    child.parentNode = null;
    var doc = this.ownerDocument;
    if (doc.activeElement !== null && child.contains(doc.activeElement)) {
      doc.activeElement = null;
    }
    return child;
  };

  Element.prototype.contains = function (node) {
    for (; node; node = node.parentNode) {
      if (node === this) {
        return true;
      }
    }
    return false;
  };

  Element.prototype.setAttribute = function (name, value) {
    var value_ = String(value);
    for (var i = 0; i < this.attributes.length; i++) {
      if (this.attributes[i].name === name) {
        this.attributes[i].value = value_;
        return;
      }
    }
    this.attributes.push({ name: name, value: value_ });
  };

  Element.prototype.getAttribute = function (name) {
    for (var i = 0; i < this.attributes.length; i++) {
      if (this.attributes[i].name === name) {
        return this.attributes[i].value;
      }
    }
    return null;
  };

  Element.prototype.removeAttribute = function (name) {
    this.attributes = this.attributes.filter(function (attribute) {
      return attribute.name !== name;
    });
  };

  // An element's text, all of it; and, set, the element's only child.
  Object.defineProperty(Element.prototype, "textContent", {
    get: function () {
      return this.childNodes
        .map(function (node) {
          return node.nodeType === 3 ? node.data : node.textContent;
        })
        .join("");
    },
    set: function (text) {
      var element = this;
      this.childNodes.slice().forEach(function (node) {
        element.removeChild(node);
      });
      if (String(text) !== "") {
        this.appendChild(new Text(this.ownerDocument, text));
      }
    }
  });

  // Scrolls the element into view: here, notes that it was, and where to,
  // as `block` says.
  Element.prototype.scrollIntoView = function (options) {
    this.ownerDocument.revealed.push({ element: this, block: options && options.block });
  };

  // A details element's `open`, which its attribute holds. When it changes,
  // the element fires toggle, as a browser does; at once, where a browser
  // fires it a little later.
  Object.defineProperty(Element.prototype, "open", {
    get: function () {
      return this.getAttribute("open") !== null;
    },
    set: function (open) {
      if (Boolean(open) === this.open) {
        return;
      }
      if (open) {
        this.setAttribute("open", "");
      } else {
        this.removeAttribute("open");
      }
      this.ownerDocument.fire(this, "toggle");
    }
  });

  // An element's style: its properties, as set, in order.
  function Style() {
    this.properties = [];
  }

  Style.prototype.setProperty = function (name, value) {
    this.removeProperty(name);
    this.properties.push([name, String(value)]);
  };

  Style.prototype.removeProperty = function (name) {
    this.properties = this.properties.filter(function (property) {
      return property[0] !== name;
    });
  };

  Style.prototype.getPropertyValue = function (name) {
    for (var i = 0; i < this.properties.length; i++) {
      if (this.properties[i][0] === name) {
        return this.properties[i][1];
      }
    }
    return "";
  };

  Element.prototype.addEventListener = function (type, listener) {
    (this.listeners[type] = this.listeners[type] || []).push(listener);
  };

  return { Document: Document };
})();
