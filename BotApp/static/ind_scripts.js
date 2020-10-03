// покажем карту

var userCurrentLoc = [55.472, 37.229];
var myMap;
var multiRoute;
var dat_a;
var dat_b;
var cross_a = false;
var cross_b = false;
var freshStr;
var geolocation


document.cookie = "username=John Doe";

ymaps.ready(showmap);
// getGeolocation();

function showmap(){
    // uncomment for user geolocation
    geolocation = ymaps.geolocation;

    myMap = new ymaps.Map('map', {
        center: userCurrentLoc, // Москва
        zoom: 9
    }, {
        searchControlProvider: 'yandex#search'
    });

    // uncomment for user geolocation
    // geolocation.get({
    //     provider: 'yandex',
    //     mapStateAutoApply: true
    // }).then(function (result) {
    //     // Красным цветом пометим положение, вычисленное через ip.
    //     result.geoObjects.options.set('preset', 'islands#redCircleIcon');
    //     result.geoObjects.get(0).properties.set({
    //         balloonContentBody: 'Мое местоположение'
    //     });
    //     myMap.geoObjects.add(result.geoObjects);
    // });
}


function addRoute() {

    multiRoute = new ymaps.multiRouter.MultiRoute({
        referencePoints: [
            dat_a.coordinates,
            dat_b.coordinates
        ],
        // Routing options.
        params: {
            results: 1 // Limit on the maximum number of routes returned by the router.
        }
    }, {
        // Automatically set the map boundaries so the entire route is visible.
        boundsAutoApply: true
    });

    myMap.geoObjects.add(multiRoute);
}


// при вводе текста, мы подыскиваем варианты (через каждые 2 буквы)
function onInput(elem) {
    if (elem.id == 'address-a' && !cross_a) {showCrossA()};
    if (elem.id == 'address-b' && !cross_b) {showCrossB()};

    freshStr = document.getElementById(elem.id).value;
    if (freshStr.length % 2 === 0){
        // we pass not only the fresh string, but also whether it is address-a or address-b
        geocodeString(freshStr, elem.id);
    }
}

function showMenu() {
    var a = document.getElementById("side-menu");
    if (
    a.className ==
    "amber-mobile-menu Header__mobile-menu amber-mobile-menu_position_left amber-mobile-menu_show"
    ) {
    a.className =
        "amber-mobile-menu Header__mobile-menu amber-mobile-menu_position_left";
    } else {
    a.className =
        "amber-mobile-menu Header__mobile-menu amber-mobile-menu_position_left amber-mobile-menu_show";
    }
}

function geocodeString(str, caller_id){
    ymaps.geocode(str, {
        /**
        * Опции запроса
        * @see https://api.yandex.ru/maps/doc/jsapi/2.1/ref/reference/geocode.xml
        */
        // Сортировка результатов от центра окна карты.
        // boundedBy: myMap.getBounds(),
        // strictBounds: true,
        // Вместе с опцией boundedBy будет искать строго внутри области, указанной в boundedBy.
        // Если нужен только один результат, экономим трафик пользователей.
        results: 6,
        boundedBy: myMap.getBounds(),
        strictBounds: true
    }).then(function (res) {
        
        if (caller_id == 'address-a'){
            document.getElementById("ty_a").innerHTML = '';
            dat_a = [];
        } else {
            document.getElementById("ty_b").innerHTML = '';
            dat_b = [];
        }
          
        for (var i = 0; i <= res.geoObjects.getLength(); ++i) {
            var cObj = res.geoObjects.get(i);
            var addr = cObj.getAddressLine(),
                coords = cObj.geometry.getCoordinates(),
                bounds = cObj.properties.get('boundedBy');
                street = cObj.getThoroughfare(),
                admAr = cObj.getAdministrativeAreas().toString().replace(',', ', '),
                bldg = cObj.getPremiseNumber(),
                locality = cObj.getLocalities()[0],
                [addr1, addr2] = pasteAddress(street, admAr, bldg, locality);

            if (caller_id == 'address-a'){
                dat_a.push({address: addr, coordinates: coords});
            } else {
                dat_b.push({address: addr, coordinates: coords});
            }
            console.log(i, addr);
            var distance = ymaps.coordSystem.geo.getDistance([55.753338, 37.622078], coords)
            updateSuggestions(addr1, addr2, distance, i, caller_id);
               
            // cObj.options.set('preset', 'islands#darkBlueDotIconWithCaption');
            // cObj.properties.set('iconCaption', addr);
            // myMap.geoObjects.add(cObj);
            // myMap.setBounds(bounds, {
            //     // Проверяем наличие тайлов на данном масштабе.
            //     checkZoomRange: true
        };
    })
}

function pasteAddress(street, admAr, bldg, locality) {
    var addr1 = "";
    var addr2 = "";

    if (street != null) {
        addr1 += street;
    } 
    if (bldg != null) {
        addr1 += ', ';
        addr1 += bldg;
    }
    if (addr1 == "") {
        addr2 += admAr;
        addr1 = locality;
    } else {
        addr2 += [admAr, locality].join(', ');
    }
    
    return [addr1, addr2];
}

function onSuggSelected(elem) {

    if (elem.id.indexOf('_a') > -1){
        //destroy the suggestions
        document.getElementById("ty_a").innerHTML = '';
        idx = elem.id.split('_')[1].slice(1) // take the string from _a (excluding) onwards
        dat_a = dat_a[idx];
        //update the string in the textbox with the address selected
        document.getElementById("address-a").value = dat_a.address;
    } else {
        document.getElementById("ty_b").innerHTML = '';
        idx = elem.id.split('_')[1].slice(1) // take the string from _b (excluding) onwards
        dat_b = dat_b[idx];
        document.getElementById("address-b").value = dat_b.address;
        if (dat_a != null) {
            addRoute();
        }
    }
}

function right_icon_clicked(elem) {
    if (elem.id == 'right_icon_a'){
        document.getElementById("address-a").value = ''
        document.getElementById("ty_a").innerHTML = '';
        dat_a = null;
        if (cross_a) {
            showNavArrow(true);
        } else {
            // do somthing when the arrow is clicked
        }
    } else {
        document.getElementById("address-b").value = '';
        document.getElementById("ty_b").innerHTML = '';
        dat_b = null;
        removeCrossB();
    }
}


function updateSuggestions(addr1, addr2, distance, sugg_id, caller_id) {
    // make sure it runs ONCE
    adjust(caller_id);
    
    if (caller_id == 'address-a'){
        var popup = "ty_a";
    } else { var popup = "ty_b";}
    var ul = document.getElementById(popup);
    // ul.innerHTML = '';
    

    var li = document.createElement("li");
    if (caller_id == 'address-a'){
        li.setAttribute("id", "suggestion_a".concat(sugg_id));
    } else {
        li.setAttribute("id", "suggestion_b".concat(sugg_id));
    }
    li.setAttribute("onclick", "onSuggSelected(this)");
    li.setAttribute(
    "class",
    "amber-list-item amber-list-item_interactive Suggest__item"
    );

    var div0 = document.createElement("div");
    div0.setAttribute(
    "class",
    "amber-list-item__col amber-list-item__col_grow"
    );
    var div01 = document.createElement("div");
    div01.setAttribute("class", "amber-list-item-content");
    var div010 = document.createElement("div");
    div010.setAttribute("class", "amber-list-item-content__title");
    var div011 = document.createElement("div");
    div011.setAttribute(
    "class",
    "amber-list-item-content__description"
    );
    var div1 = document.createElement("div");
    div1.setAttribute(
    "class",
    "amber-list-item__col Suggest__distance"
    );
    
    highLightedPart = freshStr.split(' ')[0];
    var idx = findBold(addr1, highLightedPart);
    var _unbold = document.createTextNode(addr1.slice(0, idx));
    var unbold_ = document.createTextNode(addr1.slice(idx+highLightedPart.length, addr1.length));
    var bold = document.createElement("strong");
    bold.appendChild(document.createTextNode(addr1.slice(idx, idx+highLightedPart.length)));
    div010.appendChild(_unbold);
    div010.appendChild(bold);
    div010.appendChild(unbold_);

    div011.appendChild(document.createTextNode(addr2));
    div1.appendChild(document.createTextNode((distance/1000).toFixed(2).toString() + " км"));

    div0.appendChild(div01);
    div01.appendChild(div010);
    div01.appendChild(div011);
    li.appendChild(div0);
    li.appendChild(div1);

    ul.appendChild(li);
}

function findBold(addr1, highLightedPart){
    var idx = addr1.toLowerCase().indexOf(highLightedPart.toLowerCase());
    return idx;
}

function adjust(caller_id){
    var bodyRect = document.body.getBoundingClientRect();
    var elemRect = document.getElementById(caller_id).getBoundingClientRect();
    var top_offset   = elemRect.top - bodyRect.top + elemRect.height;
    var left_offset   = elemRect.left - bodyRect.left;
    var width = elemRect.width;

    if (caller_id == 'address-a'){
        var popup = "address-a-popup";
    } else { var popup = "address-b-popup";}
    document.getElementById(popup).style.transform = 'translate('+ left_offset + 'px, ' + top_offset + 'px)';
    document.getElementById(popup).style.width = width+'px';
}

function geoLocSuccess(pos) {
    userCurrentLoc = [pos.coords.latitude, pos.coords.longitude];
    myMap.panTo(userCurrentLoc, {duration: 2000});
}

function getGeolocation() {
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(
            geoLocSuccess,
            function(){console.log('error in geolocation')}
            );
    }
    else{alert('errror in else');}
}

function showNavArrow(a) {
    if (a) {var id = 'right_icon_a';} else {var id = 'right_icon_b'};
    document.getElementById(id).innerHTML = `<span id="${id}" class="amber-input__icon amber-input__icon_right"><span class="amber-icon amber-icon_location FieldAddress__location"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24">    <path d="M12,4 C16.418,4 20,7.582 20,12 C20,16.418 16.418,20 12,20 C7.582,20 4,16.418 4,12 C4,7.582 7.582,4 12,4 Z M12,5.8 C8.581,5.8 5.8,8.581 5.8,12 C5.8,15.419 8.581,18.2 12,18.2 C15.419,18.2 18.2,15.419 18.2,12 C18.2,8.581 15.419,5.8 12,5.8 Z M15,9 L12.341,16.629 L11,13 L7.56,11.847 L15,9 Z"></path>  </svg></span></span>`;
    cross_a = false;
}

function showCrossA() {
    let id = "right_icon_a";
    document.getElementById(id).innerHTML = `<span id="${id}" onclick="right_icon_clicked(this)" class="amber-input__clear"><span class="amber-icon amber-icon_clear">  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24">    <path fill="#C4C2BE" fill-rule="evenodd" d="M13.414 12l3.536-3.536-1.414-1.414L12 10.586 8.464 7.05 7.05 8.464 10.586 12 7.05 15.536l1.414 1.414L12 13.414l3.536 3.536 1.414-1.414L13.414 12zm-7.07 5.657A8 8 0 1 1 17.656 6.343 8 8 0 0 1 6.343 17.657z">    </path>  </svg></span></span>`;
    cross_a = true;
}

function showCrossB() {
    let id = "right_icon_b";
    var node = htmlToElement(`<span id="${id}" onclick="right_icon_clicked(this)" class="amber-input__clear"><span class="amber-icon amber-icon_clear"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"><path fill="#C4C2BE" fill-rule="evenodd" d="M13.414 12l3.536-3.536-1.414-1.414L12 10.586 8.464 7.05 7.05 8.464 10.586 12 7.05 15.536l1.414 1.414L12 13.414l3.536 3.536 1.414-1.414L13.414 12zm-7.07 5.657A8 8 0 1 1 17.656 6.343 8 8 0 0 1 6.343 17.657z"></path></svg></span></span>`);
    document.getElementById('addrB').appendChild(node);
    cross_b = true;
}

function removeCrossB() {
    document.getElementById('right_icon_b').remove()
    cross_b = false;
}

function htmlToElement(html) {
    var template = document.createElement('template');
    html = html.trim(); // Never return a text node of whitespace as the result
    template.innerHTML = html;
    return template.content.firstChild;
}
