// tools/stand_in_dom.js: a stand-in DOM, for testing glem.js's view, and
// later the looks' recipes, in a JavaScript engine without a browser:
// elements with attributes, children and listeners, and the keyboard's
// focus; what the view and the tests' recipes use, and no more. It counts
// the elements that it makes, so that tests can tell which ones stay.
var standInDom = (function () {
  "use strict";

  function Document() {
    this.activeElement = null;
    this.made = 0;
  }

  Document.prototype.createElement = function (tag) {
    this.made++;
    return new Element(this, tag);
  };

  // An event, as the tests make them, on `target`: its listeners, then its
  // ancestors', as for events that bubble. `init` adds to the event, as
  // {key: "Escape"}.
  Document.prototype.fire = function (target, type, init) {
    var event = {
      type: type,
      target: target,
      defaultPrevented: false,
      preventDefault: function () {
        event.defaultPrevented = true;
      }
    };
    Object.keys(init || {}).forEach(function (name) {
      event[name] = init[name];
    });
    for (var node = target; node; node = node.parentNode) {
      (node.listeners[type] || []).slice().forEach(function (listener) {
        listener(event);
      });
    }
    return event;
  };

  // The keyboard's focus moves to an element, or away (null): focusout on
  // the one that had it, then focusin, as a browser fires them.
  Document.prototype.focus = function (element) {
    var old = this.activeElement;
    if (old === element) {
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

  function Element(doc, tag) {
    this.ownerDocument = doc;
    this.tagName = tag;
    this.parentNode = null;
    this.childNodes = [];
    // [name, value] pairs, in the order that they're first set.
    this.attributes = [];
    this.listeners = {};
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
      if (this.attributes[i][0] === name) {
        this.attributes[i][1] = value_;
        return;
      }
    }
    this.attributes.push([name, value_]);
  };

  Element.prototype.getAttribute = function (name) {
    for (var i = 0; i < this.attributes.length; i++) {
      if (this.attributes[i][0] === name) {
        return this.attributes[i][1];
      }
    }
    return null;
  };

  Element.prototype.addEventListener = function (type, listener) {
    (this.listeners[type] = this.listeners[type] || []).push(listener);
  };

  return { Document: Document };
})();
