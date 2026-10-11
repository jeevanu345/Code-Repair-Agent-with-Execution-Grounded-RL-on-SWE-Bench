/* App-owned select menus. Native controls remain the source of form values. */
(() => {
  "use strict";
  const controls = new Map();
  const descriptor = name => Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, name);
  let current = null, sequence = 0, query = "", queryTimer;
  const popup = document.createElement("div");
  popup.className = "app-select-menu";
  popup.hidden = true;
  popup.setAttribute("role", "listbox");

  function entries(select) {
    return Array.from(select.options).map((option, index) => ({option, index}))
      .filter(({option}) => !option.hidden && !option.parentElement.hidden);
  }
  function disabled(option) {
    return option.disabled || (option.parentElement.tagName === "OPTGROUP" && option.parentElement.disabled);
  }
  function label(select) {
    return select.getAttribute("aria-label") || Array.from(select.labels || [])
      .map(item => (item.querySelector(".form-label")?.textContent || item.firstChild?.textContent || "").trim())
      .filter(Boolean).join(" ") || select.title || "Choose an option";
  }
  function sync(control) {
    const {select, button, text} = control;
    text.textContent = Array.from(select.selectedOptions).map(option => option.label).join(", ") || "Choose an option";
    control.wrapper.hidden = select.hidden;
    button.disabled = select.matches(":disabled");
    button.title = select.title;
    button.setAttribute("aria-required", String(select.required));
    button.setAttribute("aria-label", `${label(select)}: ${text.textContent}`);
    for (const name of ["aria-describedby", "aria-labelledby"]) {
      const value = select.getAttribute(name);
      if (value) button.setAttribute(name, value); else button.removeAttribute(name);
    }
    if (select.getAttribute("aria-invalid")) button.setAttribute("aria-invalid", select.getAttribute("aria-invalid"));
    else if (select.validity.valid) button.removeAttribute("aria-invalid");
    if (current?.control === control) {
      if (button.disabled || !select.isConnected) close();
      else render();
    }
  }
  function close(focus = false) {
    if (!current) return;
    const {control} = current;
    control.button.setAttribute("aria-expanded", "false");
    control.button.removeAttribute("aria-activedescendant");
    popup.hidden = true;
    popup.replaceChildren();
    current = null;
    query = "";
    clearTimeout(queryTimer);
    if (focus && control.button.isConnected) control.button.focus();
  }
  function position() {
    if (!current) return;
    const rect = current.control.button.getBoundingClientRect();
    const viewport = window.visualViewport;
    const height = viewport?.height || window.innerHeight;
    const width = viewport?.width || window.innerWidth;
    const topOffset = viewport?.offsetTop || 0;
    const leftOffset = viewport?.offsetLeft || 0;
    const below = height + topOffset - rect.bottom - 8;
    const above = rect.top - topOffset - 8;
    const upward = below < Math.min(280, popup.scrollHeight) && above > below;
    const menuHeight = Math.max(40, Math.min(320, upward ? above : below));
    popup.style.maxHeight = `${menuHeight}px`;
    popup.style.width = `${Math.min(Math.max(rect.width, 180), width - 16)}px`;
    popup.style.left = `${Math.max(leftOffset + 8, Math.min(rect.left, leftOffset + width - parseFloat(popup.style.width) - 8))}px`;
    popup.style.top = `${upward ? Math.max(topOffset + 8, rect.top - Math.min(popup.scrollHeight, menuHeight) - 4) : rect.bottom + 4}px`;
  }
  function render() {
    if (!current) return;
    const {control} = current;
    const items = entries(control.select);
    if (!items.some(item => item.index === current.active && !disabled(item.option))) {
      current.active = items.find(item => !disabled(item.option))?.index ?? -1;
    }
    const nodes = [];
    let group = null;
    for (const {option, index} of items) {
      const parent = option.parentElement;
      if (parent.tagName === "OPTGROUP" && parent !== group) {
        const heading = document.createElement("div");
        heading.className = "app-select-group";
        heading.textContent = parent.label;
        heading.setAttribute("role", "presentation");
        nodes.push(heading);
      }
      group = parent;
      const item = document.createElement("div");
      item.className = "app-select-option";
      item.id = `${control.id}-option-${index}`;
      item.dataset.index = String(index);
      item.setAttribute("role", "option");
      item.setAttribute("aria-selected", String(option.selected));
      item.setAttribute("aria-disabled", String(disabled(option)));
      item.classList.toggle("is-active", index === current.active);
      const check = document.createElement("span");
      check.className = "app-select-check";
      check.setAttribute("aria-hidden", "true");
      check.textContent = option.selected ? "✓" : "";
      const text = document.createElement("span");
      text.textContent = option.label;
      item.append(check, text);
      nodes.push(item);
    }
    if (!nodes.length) {
      const empty = document.createElement("div");
      empty.className = "app-select-empty";
      empty.textContent = "No options available";
      nodes.push(empty);
    }
    popup.replaceChildren(...nodes);
    control.button.setAttribute("aria-activedescendant", current.active < 0 ? "" : `${control.id}-option-${current.active}`);
    position();
    popup.querySelector(".is-active")?.scrollIntoView({block: "nearest"});
  }
  function open(control, direction = 1) {
    if (control.select.matches(":disabled")) return;
    close();
    const items = entries(control.select).filter(item => !disabled(item.option));
    const selected = items.find(item => item.option.selected);
    current = {control, active: selected?.index ?? (direction < 0 ? items.at(-1)?.index : items[0]?.index) ?? -1};
    popup.id = `${control.id}-listbox`;
    popup.setAttribute("aria-label", label(control.select));
    popup.setAttribute("aria-multiselectable", String(control.select.multiple));
    popup.hidden = false;
    control.button.setAttribute("aria-expanded", "true");
    render();
  }
  function choose(index) {
    if (!current) return;
    const {control} = current;
    const select = control.select, option = select.options[index];
    if (!option || disabled(option)) return;
    if (select.multiple) option.selected = !option.selected;
    else select.selectedIndex = index;
    select.dispatchEvent(new Event("input", {bubbles: true}));
    select.dispatchEvent(new Event("change", {bubbles: true}));
    sync(control);
    if (!select.multiple) close(true);
  }
  function keys(event, control) {
    if (event.ctrlKey || event.metaKey) return;
    if (event.key === "Tab") { close(); return; }
    if (event.key === "Escape") {
      if (current?.control === control) {event.preventDefault(); event.stopPropagation(); close(true);}
      return;
    }
    if (["ArrowDown", "ArrowUp", "Home", "End", "Enter", " "].includes(event.key) || event.key.length === 1) {
      event.stopPropagation();
      event.preventDefault();
    } else return;
    if (current?.control !== control) {
      open(control, event.key === "ArrowUp" ? -1 : 1);
      if (["Enter", " ", "ArrowDown", "ArrowUp"].includes(event.key)) return;
    }
    const items = entries(control.select).filter(item => !disabled(item.option));
    if (["Enter", " "].includes(event.key) && !query) {choose(current.active); return;}
    let target;
    if (event.key === "Home") target = items[0];
    else if (event.key === "End") target = items.at(-1);
    else if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      const index = items.findIndex(item => item.index === current.active);
      target = items[Math.max(0, Math.min(items.length - 1, index + (event.key === "ArrowDown" ? 1 : -1)))];
    } else {
      clearTimeout(queryTimer);
      query += event.key.toLocaleLowerCase();
      queryTimer = setTimeout(() => {query = "";}, 700);
      const start = items.findIndex(item => item.index === current.active);
      const ordered = [...items.slice(start + 1), ...items.slice(0, start + 1)];
      target = ordered.find(item => item.option.label.toLocaleLowerCase().startsWith(query));
      if (!target && new Set(query).size === 1) target = ordered.find(item => item.option.label.toLocaleLowerCase().startsWith(event.key.toLocaleLowerCase()));
    }
    if (target) {current.active = target.index; render();}
  }
  function enhance(select) {
    if (controls.has(select)) return;
    const id = `app-select-${++sequence}`;
    const wrapper = document.createElement("span");
    wrapper.className = "app-select";
    const button = document.createElement("button");
    button.type = "button";
    button.className = `${select.className} app-select-trigger`;
    button.id = `${id}-trigger`;
    button.tabIndex = select.tabIndex;
    button.setAttribute("role", "combobox");
    button.setAttribute("aria-haspopup", "listbox");
    button.setAttribute("aria-expanded", "false");
    button.setAttribute("aria-controls", `${id}-listbox`);
    const text = document.createElement("span");
    text.className = "app-select-value";
    const arrow = document.createElement("span");
    arrow.className = "app-select-arrow";
    arrow.setAttribute("aria-hidden", "true");
    arrow.textContent = "⌄";
    button.append(text, arrow);
    select.before(wrapper);
    wrapper.append(select, button);
    select.classList.add("app-select-native");
    select.tabIndex = -1;
    select.setAttribute("aria-hidden", "true");
    const control = {select, wrapper, button, text, id};
    controls.set(select, control);
    // Preserve existing form code, including assignments without change events.
    for (const name of ["value", "selectedIndex"]) {
      const property = descriptor(name);
      Object.defineProperty(select, name, {
        configurable: true,
        get() { return property.get.call(this); },
        set(value) { property.set.call(this, value); sync(control); }
      });
    }
    button.addEventListener("click", () => current?.control === control ? close() : open(control));
    button.addEventListener("keydown", event => keys(event, control));
    select.addEventListener("change", () => sync(control));
    select.addEventListener("focus", () => button.focus());
    select.addEventListener("invalid", event => {
      event.preventDefault();
      button.setAttribute("aria-invalid", "true");
      button.focus();
    });
    sync(control);
  }
  function scan() {
    for (const [select, control] of controls) {
      if (!select.isConnected) {
        if (current?.control === control) close();
        controls.delete(select);
      }
    }
    document.querySelectorAll("select").forEach(enhance);
  }
  function init() {
    document.body.append(popup);
    scan();
    new MutationObserver(records => {
      let added = false;
      const updated = new Set();
      for (const record of records) {
        if (record.type === "childList" && (record.addedNodes.length || record.removedNodes.length)) added = true;
        const select = record.target instanceof Element ? record.target.closest("select") : record.target.parentElement?.closest("select");
        if (select && controls.has(select)) updated.add(controls.get(select));
        if (record.target instanceof HTMLFieldSetElement) record.target.querySelectorAll("select").forEach(select => {if (controls.has(select)) updated.add(controls.get(select));});
      }
      if (added) scan();
      updated.forEach(sync);
    }).observe(document.body, {subtree: true, childList: true, characterData: true, attributes: true, attributeFilter: ["disabled", "selected", "label", "hidden", "required", "aria-label", "aria-labelledby", "aria-describedby", "aria-invalid"]});
    popup.addEventListener("pointerdown", event => event.preventDefault());
    popup.addEventListener("click", event => {
      const option = event.target.closest(".app-select-option");
      if (option) choose(Number(option.dataset.index));
    });
    document.addEventListener("pointerdown", event => {
      if (current && !popup.contains(event.target) && !current.control.wrapper.contains(event.target)) close();
    });
    document.addEventListener("focusin", event => {
      if (current && event.target !== current.control.button && !popup.contains(event.target)) close();
    });
    document.addEventListener("click", event => {
      const label = event.target.closest("label");
      if (label?.control instanceof HTMLSelectElement && controls.has(label.control) && !event.target.closest("button")) {
        event.preventDefault();
        controls.get(label.control).button.focus();
      }
    });
    document.addEventListener("reset", event => setTimeout(() => {
      event.target.querySelectorAll("select").forEach(select => {if (controls.has(select)) sync(controls.get(select));});
      close();
    }, 0));
    document.addEventListener("scroll", event => {if (current && !popup.contains(event.target)) position();}, true);
    window.addEventListener("resize", () => close());
    window.visualViewport?.addEventListener("resize", () => close());
    window.visualViewport?.addEventListener("scroll", position);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init, {once: true});
  else init();
})();
