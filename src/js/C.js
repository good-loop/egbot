import Enum from 'easy-enums';
import Roles from './base/Roles';
import C from './base/CBase';

export default C;

/**
 * app config
 */
C.app = {
	name: "EgBot",
	service: "egbot"
};

C.TYPES = new Enum("");

C.ROLES = new Enum("user admin");
C.CAN = new Enum("view edit admin sudo");
// setup roles
Roles.defineRole(C.ROLES.user, [C.CAN.view]);
Roles.defineRole(C.ROLES.admin, C.CAN.values);
