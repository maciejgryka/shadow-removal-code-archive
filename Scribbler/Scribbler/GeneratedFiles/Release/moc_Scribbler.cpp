/****************************************************************************
** Meta object code from reading C++ file 'Scribbler.h'
**
** Created: Wed 21. Nov 14:43:24 2012
**      by: The Qt Meta Object Compiler version 63 (Qt 4.8.0)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../Scribbler.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'Scribbler.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.0. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_Scribbler[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       7,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      11,   10,   10,   10, 0x08,
      18,   10,   10,   10, 0x08,
      25,   10,   10,   10, 0x08,
      36,   10,   10,   10, 0x08,
      47,   10,   10,   10, 0x08,
      59,   10,   10,   10, 0x08,
      76,   10,   10,   10, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_Scribbler[] = {
    "Scribbler\0\0open()\0undo()\0saveMask()\0"
    "loadMask()\0clearMask()\0setToolOutline()\0"
    "setToolBrush()\0"
};

void Scribbler::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        Scribbler *_t = static_cast<Scribbler *>(_o);
        switch (_id) {
        case 0: _t->open(); break;
        case 1: _t->undo(); break;
        case 2: _t->saveMask(); break;
        case 3: _t->loadMask(); break;
        case 4: _t->clearMask(); break;
        case 5: _t->setToolOutline(); break;
        case 6: _t->setToolBrush(); break;
        default: ;
        }
    }
    Q_UNUSED(_a);
}

const QMetaObjectExtraData Scribbler::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject Scribbler::staticMetaObject = {
    { &QMainWindow::staticMetaObject, qt_meta_stringdata_Scribbler,
      qt_meta_data_Scribbler, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &Scribbler::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *Scribbler::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *Scribbler::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_Scribbler))
        return static_cast<void*>(const_cast< Scribbler*>(this));
    return QMainWindow::qt_metacast(_clname);
}

int Scribbler::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QMainWindow::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 7)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 7;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
